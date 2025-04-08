import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File } from "parse-diff";
import minimatch from "minimatch";
import { Buffer } from "buffer"; // Needed for base64 decoding

// --- Configuration ---
const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");
// New input: Maximum file size in bytes to process (e.g., 100KB)
// Avoids sending excessively large files to the AI
const MAX_FILE_SIZE_BYTES: number = parseInt(core.getInput("MAX_FILE_SIZE_BYTES") || "100000", 10);

// --- Octokit and OpenAI Clients ---
const octokit = new Octokit({ auth: GITHUB_TOKEN });
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

// --- Interfaces ---
interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
  head_sha: string; // Add head SHA to fetch file content
}

// --- Functions ---

/**
 * Fetches details about the pull request, including the head SHA.
 */
async function getPRDetails(): Promise<PRDetails> {
  const eventPath = process.env.GITHUB_EVENT_PATH;
  if (!eventPath) {
    throw new Error("GITHUB_EVENT_PATH environment variable not set.");
  }
  const eventPayload = JSON.parse(readFileSync(eventPath, "utf8"));

  const repository = eventPayload.repository;
  const number = eventPayload.pull_request?.number || eventPayload.number; // Handle different event types

  if (!number) {
     throw new Error("Could not determine pull request number from event payload.");
  }

  if (!repository) {
      throw new Error("Could not determine repository from event payload.");
  }

  const owner = repository.owner.login;
  const repo = repository.name;

  const prResponse = await octokit.pulls.get({
    owner,
    repo,
    pull_number: number,
  });

  return {
    owner,
    repo,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
    head_sha: prResponse.data.head.sha, // Get the head SHA
  };
}

/**
 * Fetches the diff for a given pull request.
 */
async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  try {
    const response = await octokit.pulls.get({
      owner,
      repo,
      pull_number,
      mediaType: { format: "diff" },
    });
    // The data is expected to be a string for the diff format
    // Using type assertion as Octokit's types might be broad
    return response.data as unknown as string;
  } catch (error) {
    core.error(`Error fetching diff for PR #${pull_number}: ${error}`);
    return null;
  }
}

/**
 * Fetches the full content of a file at a specific commit SHA.
 */
async function getFileContent(
    owner: string,
    repo: string,
    filePath: string,
    sha: string
): Promise<string | null> {
    try {
        const response = await octokit.repos.getContent({
            owner,
            repo,
            path: filePath,
            ref: sha,
        });

        // Check if the response data is for a file and has content
        if (response.data && 'type' in response.data && response.data.type === 'file' && response.data.content) {
            if (response.data.encoding === 'base64') {
                return Buffer.from(response.data.content, 'base64').toString('utf8');
            } else {
                core.warning(`Unexpected encoding for file ${filePath}: ${response.data.encoding}`);
                // Attempt to return content directly if not base64 (though usually it is)
                 return response.data.content;
            }
        } else if (response.data && 'type' in response.data && response.data.type !== 'file') {
             core.warning(`Path ${filePath} is not a file (type: ${response.data.type}). Skipping content fetch.`);
             return null;
        } else {
            core.warning(`Could not retrieve content for file ${filePath}. Response data might be missing expected properties.`);
            return null;
        }
    } catch (error: any) {
        // Handle common errors like file not found (404) gracefully
        if (error.status === 404) {
             core.warning(`File ${filePath} not found at SHA ${sha}. It might have been deleted or renamed.`);
        } else {
             core.error(`Error fetching content for file ${filePath} at SHA ${sha}: ${error}`);
        }
        return null;
    }
}


/**
 * Analyzes code changes using AI, providing full file context.
 */
async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const file of parsedDiff) {
    // Skip deleted files and files that are directories in the diff
    if (file.to === "/dev/null" || file.to === undefined || file.binary) {
        core.info(`Skipping deleted, binary, or invalid file entry: ${file.from ?? file.to}`);
        continue;
    }

    // --- Fetch Full File Content ---
    core.info(`Processing file: ${file.to}`);
    let fullFileContent: string | null = null;
    try {
        // Check file size before fetching (using diff stats if available, otherwise fetch metadata)
        // Note: file.additions/deletions might not accurately reflect final size.
        // A more accurate check might involve a HEAD request or preliminary getContent if needed.
        // For simplicity, we'll fetch and then check size here.

        fullFileContent = await getFileContent(prDetails.owner, prDetails.repo, file.to, prDetails.head_sha);

        if (fullFileContent === null) {
             core.warning(`Could not fetch content for ${file.to}. Skipping analysis for this file.`);
             continue; // Skip analysis if content couldn't be fetched
        }

        // Check file size
        const fileSize = Buffer.byteLength(fullFileContent, 'utf8');
        if (fileSize > MAX_FILE_SIZE_BYTES) {
            core.warning(`File ${file.to} is too large (${fileSize} bytes > ${MAX_FILE_SIZE_BYTES} bytes). Skipping AI analysis.`);
            continue; // Skip analysis for this file
        }

    } catch (error) {
         core.error(`Error fetching or checking size for file ${file.to}: ${error}`);
         continue; // Skip analysis for this file on error
    }
    // --- End Fetch Full File Content ---


    // Iterate through chunks to provide diff context alongside full file
    for (const chunk of file.chunks) {
      // --- Create Prompt with Full File Context ---
      const prompt = createPrompt(file, chunk, prDetails, fullFileContent); // Pass full content
      // --- Get AI Response ---
      const aiResponse = await getAIResponse(prompt);
      if (aiResponse) {
        // --- Format Comments ---
        // Pass the full file content to potentially help map line numbers if needed,
        // although the AI is expected to return line numbers relative to the full file.
        const newComments = createComment(file, chunk, aiResponse, fullFileContent);
        if (newComments && newComments.length > 0) {
          comments.push(...newComments);
        }
      }
    }
  }
  return comments;
}

/**
 * Creates the prompt for the AI, including full file content and diff chunk.
 */
function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails, fullFileContent: string): string {
  // Construct the diff snippet for context
  const diffSnippet = `
${chunk.content}
${chunk.changes
    // @ts-expect-error - ln and ln2 exist where needed based on diff format
    .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
    .join("\n")}
`;

  return `Your task is to review pull requests. You will be given the full content of a file and a specific diff chunk from a pull request.
Instructions:
- Review the **entire file content** provided below for potential bugs, style issues, performance concerns, or areas for improvement.
- Pay close attention to the area highlighted by the **Git diff to review** section, as this indicates the most recent changes. Focus your comments primarily on or near these changed lines, but consider their impact in the context of the whole file.
- Provide the response in the following JSON format: {"reviews": [{"lineNumber": <line_number_in_full_file>, "reviewComment": "<review comment>"}]}
- The "lineNumber" MUST correspond to the line number in the **full file content**.
- Do not give positive comments or compliments.
- Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Use the PR title and description for overall context but focus comments on the code itself.
- IMPORTANT: NEVER suggest adding comments to the code.

---
Pull Request Context:
File Path: ${file.to}
Pull Request Title: ${prDetails.title}
Pull Request Description:
${prDetails.description}
---
Full File Content to Review:
\`\`\`${file.to?.split('.').pop() || ''}
${fullFileContent}
\`\`\`
---
Git diff to review (for focusing comments):
\`\`\`diff
${diffSnippet}
\`\`\`
---
JSON Response:`;
}

/**
 * Gets the AI response for a given prompt.
 */
async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string; // Keep as string initially, parse later
  reviewComment: string;
}> | null> {
  const queryConfig = {
    model: OPENAI_API_MODEL,
    temperature: 0.2,
    max_tokens: 1000, // Increased slightly, adjust as needed
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  };

  try {
    // Check if the model supports JSON response format
    const supportsJsonFormat = [
        "gpt-4-1106-preview", // Older turbo preview
        "gpt-4-0125-preview", // Newer turbo preview
        "gpt-4-turbo-preview", // Alias
        "gpt-4-turbo",         // Latest GPT-4 Turbo
        "gpt-3.5-turbo-1106", // Older 3.5 turbo supporting JSON mode
        "gpt-3.5-turbo-0125", // Newer 3.5 turbo
    ].includes(OPENAI_API_MODEL);

    const response = await openai.chat.completions.create({
      ...queryConfig,
      // Conditionally add response_format for supported models
      ...(supportsJsonFormat ? { response_format: { type: "json_object" } } : {}),
      messages: [
        {
          role: "system",
          content: prompt, // The detailed prompt including full file and diff
        },
      ],
    });

    const rawResponse = response.choices[0].message?.content?.trim();

    if (!rawResponse) {
        core.warning("AI response content is empty.");
        return null;
    }

    // Attempt to parse the response as JSON
    try {
        // Sometimes the response might be wrapped in ```json ... ```, try to extract
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/);
        const jsonString = jsonMatch ? jsonMatch[1] : rawResponse;
        const parsedResponse = JSON.parse(jsonString);

        // Validate the structure
        if (parsedResponse && Array.isArray(parsedResponse.reviews)) {
             // Further validation for each review item (optional but recommended)
             const validReviews = parsedResponse.reviews.filter((review: any) =>
                typeof review.lineNumber === 'string' || typeof review.lineNumber === 'number' && // Allow number too
                typeof review.reviewComment === 'string' &&
                review.reviewComment.trim() !== ''
             );
             // Ensure lineNumber is string for consistency before returning
             validReviews.forEach((review: any) => review.lineNumber = String(review.lineNumber));
             return validReviews;
        } else {
             core.warning(`AI response is not in the expected JSON format: ${rawResponse}`);
             return null;
        }
    } catch (parseError) {
         core.error(`Failed to parse AI JSON response: ${parseError}\nRaw response: ${rawResponse}`);
         return null;
    }

  } catch (error) {
    core.error(`Error getting AI response: ${error}`);
    return null;
  }
}

/**
 * Formats AI responses into GitHub comment objects.
 */
function createComment(
  file: File,
  chunk: Chunk, // Chunk might still be useful for context or future refinement
  aiResponses: Array<{
    lineNumber: string; // Comes as string from getAIResponse
    reviewComment: string;
  }>,
  fullFileContent: string // Pass full content for potential future use (e.g., line validation)
): Array<{ body: string; path: string; line: number }> {
    if (!file.to) {
        core.warning("File path (file.to) is missing, cannot create comments.");
        return [];
    }
    const filePath = file.to; // Use the correct file path

    return aiResponses.flatMap((aiResponse) => {
        const lineNumber = parseInt(aiResponse.lineNumber, 10);

        // Basic validation for line number
        if (isNaN(lineNumber) || lineNumber <= 0) {
            core.warning(`Invalid line number '${aiResponse.lineNumber}' received from AI for file ${filePath}. Skipping comment.`);
            return []; // Return empty array for flatMap to ignore this item
        }

        // Optional: Add more validation, e.g., check if lineNumber exceeds lines in fullFileContent

        return {
            body: aiResponse.reviewComment,
            path: filePath,
            line: lineNumber, // Use the line number relative to the full file
        };
    });
}

/**
 * Creates review comments on the pull request.
 */
async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {
  if (comments.length === 0) {
    core.info("No comments to post.");
    return;
  }
  core.info(`Posting ${comments.length} comments to PR #${pull_number}...`);
  try {
    await octokit.pulls.createReview({
      owner,
      repo,
      pull_number,
      comments,
      event: "COMMENT", // Post comments without approving/requesting changes
    });
     core.info("Successfully posted comments.");
  } catch (error) {
     core.error(`Failed to create review comments: ${error}`);
     // Consider logging the comments that failed to post for debugging
     // core.error(`Failed comments data: ${JSON.stringify(comments)}`);
  }
}

// --- Main Execution ---
async function main() {
  try {
    const prDetails = await getPRDetails();
    core.info(`Processing PR #${prDetails.pull_number} in ${prDetails.owner}/${prDetails.repo}`);

    let diff: string | null;
    const eventPath = process.env.GITHUB_EVENT_PATH;
     if (!eventPath) {
        throw new Error("GITHUB_EVENT_PATH environment variable not set.");
     }
    const eventData = JSON.parse(readFileSync(eventPath, "utf8"));

    // Determine how to get the diff based on the event action
    if (eventData.action === "opened" || eventData.action === "reopened") {
      core.info("Action is 'opened' or 'reopened', fetching full PR diff.");
      diff = await getDiff(
        prDetails.owner,
        prDetails.repo,
        prDetails.pull_number
      );
    } else if (eventData.action === "synchronize") {
      core.info("Action is 'synchronize', fetching diff between commits.");
      const baseSha = eventData.before;
      const headSha = eventData.after; // Note: prDetails.head_sha should match this
       if (!baseSha || !headSha) {
            core.error("Could not determine base or head SHA for 'synchronize' event.");
            return;
       }
      core.info(`Comparing base ${baseSha} to head ${headSha}`);
      const response = await octokit.repos.compareCommits({
        headers: {
          accept: "application/vnd.github.v3.diff", // Request diff format
        },
        owner: prDetails.owner,
        repo: prDetails.repo,
        base: baseSha,
        head: headSha,
      });
      diff = String(response.data); // Diff content is in response.data
    } else {
      core.warning(`Unsupported event action: '${eventData.action}'. Skipping review.`);
      return;
    }

    if (!diff) {
      core.info("No diff content found or fetched. Nothing to review.");
      return;
    }

    // Parse the diff
    const parsedDiff = parseDiff(diff);

    // Get exclude patterns from input
    const excludePatterns = core
      .getInput("exclude")
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0); // Filter out empty strings

    // Filter files based on exclude patterns
    const filteredDiff = parsedDiff.filter((file) => {
       if (!file.to) return false; // Skip if no destination file path
       const shouldExclude = excludePatterns.some((pattern) =>
         minimatch(file.to ?? "", pattern)
       );
       if (shouldExclude) {
           core.info(`Excluding file ${file.to} due to pattern match.`);
       }
       return !shouldExclude;
    });

    if (filteredDiff.length === 0) {
        core.info("All files in the diff were excluded or no files to process. Nothing to review.");
        return;
    }

    // Analyze the code (providing full file context now)
    const comments = await analyzeCode(filteredDiff, prDetails);

    // Create the review comment on GitHub
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );

    core.info("AI review process completed.");

  } catch (error) {
    core.setFailed(`Action failed with error: ${error instanceof Error ? error.message : error}`);
     if (error instanceof Error && error.stack) {
         core.error(error.stack);
     }
  }
}

// Run the main function
main();

