const sharp = require("sharp");

async function renderSvg(input) {
  const output = `${input}.png`;
  const info = await sharp(input, { density: 220 })
    .flatten({ background: "#ffffff" })
    .png()
    .toFile(output);
  console.log(`${output} ${info.width}x${info.height}`);
}

async function main() {
  const inputs = process.argv.slice(2);
  if (inputs.length === 0) {
    console.error("usage: node scripts/render_svg_pngs.js figure.svg [...]");
    process.exit(2);
  }
  for (const input of inputs) {
    await renderSvg(input);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
