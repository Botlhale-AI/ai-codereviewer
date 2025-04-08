import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File, Change } from "parse-diff"; // Import Change type
import minimatch from "minimatch";
import { Buffer } from "buffer"; // Needed for base64 decoding

// --- Configuration ---
const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");
const MAX_FILE_SIZE_BYTES: number = parseInt(core.getInput("MAX_FILE_SIZE_BYTES") || "100000", 10);
const excludePatterns = core
  .getInput("exclude")
  .split(",")
  .map((s) => s.trim())
  .filter((s) => s.length > 0);

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
  head_sha: string;
}

interface ReviewComment {
  body: string;
  path: string;
  line: number;
}

interface AIResponseItem {
  lineNumber: string; // Keep as string initially, parse later
  reviewComment: string;
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
  const number = eventPayload.pull_request?.number || eventPayload.number;

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
    head_sha: prResponse.data.head.sha,
  };
}

/**
 * Fetches the diff for a given pull request or commit range.
 */
async function getDiffContent(prDetails: PRDetails, eventData: any): Promise<string | null> {
  let diff: string | null = null;
  // Handle 'pull_request' events (opened, reopened, synchronize)
  if (eventData.pull_request) {
     const pull_number = eventData.pull_request.number;
     if (eventData.action === "opened" || eventData.action === "reopened") {
         core.info(`Workspaceing full PR diff for 'opened'/'reopened' event (PR #${pull_number}).`);
         try {
            const response = await octokit.pulls.get({
                owner: prDetails.owner,
                repo: prDetails.repo,
                pull_number: pull_number,
                mediaType: { format: "diff" },
            });
            diff = response.data as unknown as string;
         } catch (error) {
            core.error(`Error fetching full PR diff: ${error}`);
         }
     } else if (eventData.action === "synchronize") {
        core.info(`Workspaceing diff between commits for 'synchronize' event (PR #${pull_number}).`);
        const baseSha = eventData.before;
        const headSha = eventData.after;
        if (!baseSha || !headSha) {
            core.error("Could not determine base or head SHA for 'synchronize' event.");
        } else {
            core.info(`Comparing base ${baseSha} to head ${headSha}`);
            try {
                const response = await octokit.repos.compareCommits({
                headers: { accept: "application/vnd.github.v3.diff" },
                owner: prDetails.owner,
                repo: prDetails.repo,
                base: baseSha,
                head: headSha,
                });
                diff = String(response.data);
            } catch (error) {
                core.error(`Error comparing commits: ${error}`);
            }
        }
     } else {
        core.warning(`Unsupported pull_request event action: '${eventData.action}'. Cannot determine diff.`);
     }
  } else {
     // Fallback or handle other event types if necessary (e.g., push)
     core.warning(`Event is not a pull_request event. Diff fetching might be inaccurate or skipped.`);
     // Optionally try fetching full PR diff if prDetails are available anyway
     // (May not be the correct diff for other event types)
      try {
         core.info(`Attempting to fetch full PR diff as fallback for PR #${prDetails.pull_number}`);
         const response = await octokit.pulls.get({
             owner: prDetails.owner,
             repo: prDetails.repo,
             pull_number: prDetails.pull_number,
             mediaType: { format: "diff" },
         });
         diff = response.data as unknown as string;
      } catch (error) {
         core.error(`Error fetching full PR diff as fallback: ${error}`);
      }
  }
  return diff;
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

        // Type guard to check if response.data is an object for a file
        if (typeof response.data === 'object' && response.data && 'type' in response.data && response.data.type === 'file' && 'content' in response.data) {
            // Type guard for encoding property
            if ('encoding' in response.data && response.data.encoding === 'base64') {
                return Buffer.from(response.data.content, 'base64').toString('utf8');
            } else {
                core.warning(`Unexpected or missing encoding for file ${filePath}. Assuming UTF-8.`);
                 if (response.data.content) {
                    try {
                       return Buffer.from(response.data.content, 'base64').toString('utf8');
                    } catch (e) {
                       return typeof response.data.content === 'string' ? response.data.content : null;
                    }
                 } else {
                     return null;
                 }
            }
        } else if (typeof response.data === 'object' && response.data && 'type' in response.data && response.data.type !== 'file') {
             core.info(`Path ${filePath} is not a file (type: ${response.data.type}). Skipping content fetch.`);
        } else if (Array.isArray(response.data)) {
            core.info(`Path ${filePath} is a directory. Skipping content fetch.`);
        } else {
            core.warning(`Could not retrieve content for file ${filePath}. Response data might be missing expected properties or was not a file.`);
        }
        return null;
    } catch (error: any) {
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
): Promise<ReviewComment[]> {
  const allComments: ReviewComment[] = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null" || !file.to || file.binary) {
      core.info(`Skipping deleted, binary, or invalid file entry: ${file.from ?? file.to ?? 'unknown'}`);
      continue;
    }
    const filePath = file.to;
    if (excludePatterns.some(pattern => minimatch(filePath, pattern))) {
        core.info(`Excluding file ${filePath} due to pattern match.`);
        continue;
    }

    core.info(`Processing file: ${filePath}`);
    let fullFileContent: string | null = null;
    try {
      fullFileContent = await getFileContent(prDetails.owner, prDetails.repo, filePath, prDetails.head_sha);

      if (fullFileContent === null) {
          core.warning(`Could not fetch content for ${filePath}. Skipping analysis for this file.`);
          continue;
      }

      const fileSize = Buffer.byteLength(fullFileContent, 'utf8');
      if (fileSize > MAX_FILE_SIZE_BYTES) {
          core.warning(`File ${filePath} is too large (${fileSize} bytes > ${MAX_FILE_SIZE_BYTES} bytes). Skipping AI analysis.`);
          continue;
      }

    } catch (error) {
       core.error(`Error fetching or checking size for file ${filePath}: ${error}`);
       continue;
    }

    const processedChunkContent = new Set<string>();

    for (const chunk of file.chunks) {
        if (processedChunkContent.has(chunk.content)) continue;
        processedChunkContent.add(chunk.content);

      const prompt = createPrompt(file, chunk, prDetails, fullFileContent);
      // Log the prompt being sent (optional, can be verbose)
      core.debug(`Prompt for chunk in ${filePath}:\n${prompt}`);

      const aiResponse = await getAIResponse(prompt);

      if (aiResponse) {
          core.debug(`Raw AI Response for chunk in ${filePath}: ${JSON.stringify(aiResponse)}`);
        const validComments = createComment(file, chunk, aiResponse);
        if (validComments.length > 0) {
          // Info log moved to createComment for more context
          allComments.push(...validComments);
        }
      }
    }
  }
  return allComments;
}

/**
 * Creates the prompt for the AI, including full file content and diff chunk.
 */
function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails, fullFileContent: string): string {
  const diffSnippet = `
${chunk.content}
${chunk.changes
    .map((c: Change) => `${c.ln ?? c.ln2 ?? ''} ${c.content}`)
    .join("\n")}
`;
  const fileExtension = file.to?.split('.').pop() || '';

  return `Your task is to review pull requests based on the provided context.
You will be given the full content of a file and a specific diff chunk from that file in a pull request.
Instructions:
1. Review the **entire file content** provided below for potential bugs, style issues (use common conventions for the language), performance concerns, security vulnerabilities, or areas for improvement.
2. Pay close attention to the area highlighted by the **Git diff chunk to review** section, as this indicates the most recent changes.
3. **Focus your comments primarily on or very near the added lines (lines starting with '+') within the diff chunk.** While you review the whole file for context, only suggest actionable changes related to the modifications in this PR.
4. Provide the response strictly in the following JSON format: {"reviews": [{"lineNumber": <line_number_in_full_file>, "reviewComment": "<review comment in GitHub Markdown>"}]}
5. The "lineNumber" MUST correspond precisely to the line number in the **full file content** where the issue is found. This line number MUST be one of the lines marked with '+' in the provided diff chunk.
6. Do not provide compliments or positive feedback. Only suggest improvements.
7. If no issues are found related to the changed lines, return an empty array: {"reviews": []}
8. Write comments concisely and clearly.
9. IMPORTANT: NEVER suggest adding comments to the code.

---
Pull Request Context:
File Path: ${file.to}
PR Title: ${prDetails.title}
PR Description: ${prDetails.description || "No description provided."}
---
Full File Content to Review (Language Hint: ${fileExtension}):
\`\`\`${fileExtension}
${fullFileContent}
\`\`\`
---
Git diff chunk to review (Focus comments on '+' lines):
\`\`\`diff
${diffSnippet}
\`\`\`
---
JSON Response (ONLY include comments for '+' lines in the diff above):`;
}


/**
 * Gets the AI response for a given prompt.
 */
async function getAIResponse(prompt: string): Promise<AIResponseItem[] | null> {
    const queryConfig = {
        model: OPENAI_API_MODEL,
        temperature: 0.2,
        max_tokens: 1000,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
    };

    try {
        const supportsJsonFormat = [
            "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview",
            "gpt-4-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125", "gpt-3.5-turbo"
        ].includes(OPENAI_API_MODEL);

        const response = await openai.chat.completions.create({
            ...queryConfig,
            ...(supportsJsonFormat ? { response_format: { type: "json_object" } } : {}),
            messages: [
                { role: "system", content: prompt },
            ],
        });

        const rawResponse = response.choices[0].message?.content?.trim();
        if (!rawResponse) {
            core.warning("AI response content is empty.");
            return null;
        }

        // Log the raw response from AI for debugging
        core.debug(`Raw AI response content:\n${rawResponse}`);

        try {
            const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/);
            const jsonString = jsonMatch ? jsonMatch[1] : rawResponse;
            const parsedResponse = JSON.parse(jsonString);

            if (parsedResponse && Array.isArray(parsedResponse.reviews)) {
                const validReviews: AIResponseItem[] = [];
                for (const review of parsedResponse.reviews) {
                    if (review &&
                        (typeof review.lineNumber === 'string' || typeof review.lineNumber === 'number') &&
                        (String(review.lineNumber)).trim() !== '' &&
                        typeof review.reviewComment === 'string' &&
                        review.reviewComment.trim() !== '')
                    {
                        validReviews.push({
                            lineNumber: String(review.lineNumber).trim(),
                            reviewComment: review.reviewComment
                        });
                    } else {
                         core.warning(`Skipping invalid review item structure: ${JSON.stringify(review)}`);
                    }
                }
                return validReviews;

            } else {
                core.warning(`AI response is not in the expected JSON format or 'reviews' is not an array: ${rawResponse}`);
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
 * Validates AI responses against the diff chunk and formats them into GitHub comment objects.
 */
function createComment(
    file: File,
    chunk: Chunk, // Needed for validation
    aiResponses: AIResponseItem[]
): ReviewComment[] {
    if (!file.to) {
        core.warning("File path (file.to) is missing, cannot create comments.");
        return [];
    }
    const filePath = file.to;

    // --- Validation Logic ---
    const addedLineNumbersInChunk = new Set<number>();
    for (const change of chunk.changes) {
        if (change.add && typeof change.ln === 'number') {
            addedLineNumbersInChunk.add(change.ln);
        }
    }
    // --- End Validation Logic ---

    // Log the chunk details for debugging validation
    core.debug(`Processing Chunk for ${filePath}. Content: ${chunk.content}`);
    core.debug(`Added lines identified in this chunk: ${JSON.stringify(Array.from(addedLineNumbersInChunk))}`);

    if (addedLineNumbersInChunk.size === 0 && aiResponses.length > 0) {
       core.info(`Chunk in ${filePath} had no added lines, but AI provided ${aiResponses.length} suggestions. Discarding line-specific comments for this chunk.`);
       aiResponses.forEach(resp => core.debug(`Discarded AI comment for ${filePath} line ${resp.lineNumber}: ${resp.reviewComment.substring(0, 50)}...`));
       return [];
    }


    return aiResponses.flatMap((aiResponse): ReviewComment[] => {
        const lineNumber = parseInt(aiResponse.lineNumber, 10);

        if (isNaN(lineNumber) || lineNumber <= 0) {
            core.warning(`Invalid line number format '${aiResponse.lineNumber}' received from AI for file ${filePath}. Skipping comment: "${aiResponse.reviewComment.substring(0, 50)}..."`);
            return [];
        }

        // --- Check if the AI's suggested line number is in the set of added lines for this chunk ---
        if (addedLineNumbersInChunk.has(lineNumber)) {
            // --- START Enhanced Logging ---
            core.info(`✔️ VALIDATING COMMENT for ${filePath} line ${lineNumber}`); // Use info level for visibility
            core.debug(`   AI Response: ${JSON.stringify(aiResponse)}`);
            core.debug(`   Added Lines in Chunk: ${JSON.stringify(Array.from(addedLineNumbersInChunk))}`);
            // Find and log the specific change object from the diff that matched
            const matchingChange = chunk.changes.find(c => c.add && c.ln === lineNumber);
            core.debug(`   Matching Diff Line Data: ${JSON.stringify(matchingChange)}`);
            // --- END Enhanced Logging ---
            return [{ // Return array with single valid comment
                body: aiResponse.reviewComment,
                path: filePath,
                line: lineNumber,
            }];
        } else {
            // --- START Enhanced Logging for Discarded Comments ---
            core.info(`❌ DISCARDING COMMENT for ${filePath} line ${lineNumber} (Not an added line in chunk)`); // Use info level
            core.debug(`   AI Response: ${JSON.stringify(aiResponse)}`);
            core.debug(`   Added Lines in Chunk: ${JSON.stringify(Array.from(addedLineNumbersInChunk))}`);
            // --- END Enhanced Logging for Discarded Comments ---
            return []; // Discard comment
        }
        // --- End Check ---
    });
}


/**
 * Creates review comments on the pull request.
 */
async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: ReviewComment[] // Use ReviewComment type
): Promise<void> {
  if (comments.length === 0) {
    core.info("No valid comments to post after validation.");
    return;
  }

  // Log the comments that are about to be posted
  core.info(`Attempting to post ${comments.length} validated comments to PR #${pull_number}...`);
  core.debug(`Comments to be posted: ${JSON.stringify(comments, null, 2)}`); // Pretty print for debug

  try {
    await octokit.pulls.createReview({
      owner,
      repo,
      pull_number,
      comments, // Already filtered to valid comments
      event: "COMMENT",
    });
    core.info("Successfully posted review comments.");
  } catch (error: any) { // Catch as 'any' or 'unknown' then check type
    core.error(`Failed to create review comments: ${error}`);
    // Log the error response details if available, especially for 422 errors
    if (error && typeof error === 'object' && 'status' in error) {
      core.error(`API Response Status: ${error.status}`);
       // Attempt to access response data if it exists (common with Octokit errors)
       if ('response' in error && error.response && typeof error.response === 'object' && 'data' in error.response) {
          core.error(`API Response Data: ${JSON.stringify(error.response.data)}`);
       }
    }
    // Also log the comments it tried to post for debugging
    core.error(`Failed comments data (first 5): ${JSON.stringify(comments.slice(0, 5), null, 2)}`);
  }
}

// --- Main Execution ---
async function main() {
  try {
    const prDetails = await getPRDetails();
    core.info(`Processing PR #${prDetails.pull_number} in ${prDetails.owner}/${prDetails.repo} (Head SHA: ${prDetails.head_sha})`);

    const eventPath = process.env.GITHUB_EVENT_PATH;
    if (!eventPath) {
      throw new Error("GITHUB_EVENT_PATH environment variable not set.");
    }
    const eventData = JSON.parse(readFileSync(eventPath, "utf8"));

    // Get diff content based on event
    const diff = await getDiffContent(prDetails, eventData);

    if (!diff) {
      core.warning("No diff content found or fetched. Nothing to review.");
      return;
    }

    // Parse the diff
    core.debug(`Raw Diff Content (first 500 chars):\n ${diff.substring(0,500)}`);
    const parsedDiff = parseDiff(diff);

    core.info(`Found ${parsedDiff.length} files in diff.`);

    // Analyze the code (validation happens inside createComment)
    const comments = await analyzeCode(parsedDiff, prDetails);

    // Create the review comment on GitHub
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments // Send only the validated comments
    );

    core.info("AI review process completed.");

  } catch (error) {
    core.setFailed(`Action failed with error: ${error instanceof Error ? error.message : String(error)}`);
    if (error instanceof Error && error.stack) {
      core.error(error.stack);
    }
  }
}

// Run the main function
main();
