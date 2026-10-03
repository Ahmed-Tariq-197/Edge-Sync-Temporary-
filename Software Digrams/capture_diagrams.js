const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

const diagramsDir = path.resolve(__dirname);

const files = [
  '1_Component_Diagram.html',
  '2_StateMachine_Diagram.html',
  '3_Sequence_Diagram_Success.html',
  '4_Sequence_Diagram_Failure.html',
  '5_Activity_Diagram.html',
  '6_Deployment_Diagram.html',
  '7_ROS2_Node_Graph.html',
  '8_Class_Diagram_Data_Model.html',
  '9_Timing_Diagram.html',
  '10_Boxes_Diagram.html',
];

function getBrowserPath() {
  const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
  if (fs.existsSync(chromePath)) return chromePath;
  if (fs.existsSync(edgePath)) return edgePath;
  throw new Error('Neither Chrome nor Edge found on system.');
}

(async () => {
  const executablePath = getBrowserPath();
  console.log(`Using browser: ${executablePath}`);

  const browser = await puppeteer.launch({
    executablePath,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-gpu',
      '--force-device-scale-factor=2',
      '--hide-scrollbars'
    ],
  });

  for (const file of files) {
    const photosDir = path.join(diagramsDir, 'Digrams Photos');
    if (!fs.existsSync(photosDir)) fs.mkdirSync(photosDir, { recursive: true });
    const pngName = file.replace('.html', '.png');
    const pngPath = path.join(photosDir, pngName);

    console.log(`\nRendering: ${file}...`);

    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 2 });

    const htmlPath = path.join(diagramsDir, file);
    const fileUrl = 'file:///' + htmlPath.replace(/\\/g, '/');
    await page.goto(fileUrl, { waitUntil: 'load', timeout: 30000 });

    // Wait for Mermaid SVGs to render
    try {
      await page.waitForFunction(() => {
        const containers = document.querySelectorAll('.diagram-container');
        if (containers.length === 0) return document.querySelectorAll('svg').length > 0;
        const rendered = Array.from(containers).filter(c => c.querySelector('svg') !== null);
        return rendered.length === containers.length;
      }, { timeout: 15000 });
      console.log(`  ✓ All SVGs rendered.`);
    } catch (e) {
      console.log(`  ⚠️ Fallback wait...`);
      await new Promise(r => setTimeout(r, 2000));
    }

    // Additional settling time
    await new Promise(r => setTimeout(r, 1200));

    // Hide print buttons before screenshot
    await page.evaluate(() => {
      const btns = document.querySelectorAll('.print-btn');
      btns.forEach(b => b.style.display = 'none');
    });

    // Capture full-page screenshot
    await page.screenshot({
      path: pngPath,
      fullPage: true,
      type: 'png'
    });

    const stats = fs.statSync(pngPath);
    console.log(`  ✅ Successfully saved: ${pngName} (${Math.round(stats.size / 1024)} KB)`);
    await page.close();
  }

  await browser.close();
  console.log('\n🎉 ALL 9 DIAGRAMS SAVED AS HIGH-RES PNG!');
})();

