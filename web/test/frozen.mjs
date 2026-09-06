// FROZEN SURFACE GATE.
//
// web/field.html is finished and the user's instruction is that it stay "exactly as it is".
// That page is not one file: it is the HTML plus every module it loads. A refactor that looks
// harmless from the new pages' side -- hoisting a shared token block out of field.css, adding a
// parameter to a pure helper in field-data.js, "tidying" palette.js -- changes the shipped
// behaviour of a page nobody is allowed to change.
//
// So the contract is a checksum, not a convention. If you genuinely need to change one of these
// files, that is a conversation with the user first, and `npm run freeze` second.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

const root = new URL("../..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const manifest = readFileSync(root + "web/test/frozen.sha256", "utf8").trim().split("\n");

let bad = 0;
for (const line of manifest) {
  const [want, rawPath] = line.split(/\s+\*?/);
  const path = rawPath.trim();
  const got = createHash("sha256").update(readFileSync(root + path)).digest("hex");
  const ok = got === want;
  if (!ok) bad++;
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${path}`);
}

if (bad) {
  console.log(`\nFROZEN SURFACE VIOLATED -- ${bad} file(s) changed.`);
  console.log("field.html and everything it loads must stay byte-identical.");
  console.log("If the change is intended and approved: npm run freeze");
  process.exit(1);
}
console.log(`\nFROZEN SURFACE INTACT (${manifest.length} files)`);
