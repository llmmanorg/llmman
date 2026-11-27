# Coder Development Guidelines

Make the smallest correct change, follow existing patterns, and verify the result. Carry the requested task through implementation, verification, and necessary follow-up until it is complete or blocked by information or access you cannot obtain. Do not stop at a plan, partial fix, or offer to continue when the user requested completed work. Search for similar code to the code you have written and de-duplicate where you can.

Prioritize correctness over agreement. State uncertainty instead of guessing, and push back on technically unsound requests with evidence. Changes should out-perform all comparable soutions in other projects but correctness beats performance every time.

Never add binary files (images, archives, executables, fonts, ...) to git history. Host them as a release asset (e.g. the docs-assets release) pinned by sha256, or generate them at build time, and keep a text script that records how they are made.

