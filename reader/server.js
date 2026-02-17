const express = require('express');
const path = require('path');
const fs = require('fs');
const os = require('os');

const app = express();
const PORT = process.env.PORT || 3002;
const OUTPUT_DIR = process.env.OUTPUT_DIR || path.resolve(__dirname, '..', 'output');

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve images from the manga output directory
app.use('/images', express.static(OUTPUT_DIR));

// Serve arbitrary files by absolute path
app.get('/file/*', (req, res) => {
  const filePath = '/' + req.params[0];
  if (!fs.existsSync(filePath)) return res.status(404).send('Not found');
  res.sendFile(filePath);
});

// API: load folder and return paired images
app.post('/api/load-folder', (req, res) => {
  let { folder, folders } = req.body;
  
  // Support single or multiple folders
  const folderList = folders ? folders : (folder ? [folder] : []);
  if (!folderList.length) return res.status(400).json({ error: 'folder(s) required' });

  const allPages = [];
  
  for (const f of folderList) {
    let dir = path.isAbsolute(f) ? f : path.join(OUTPUT_DIR, f);
    if (!fs.existsSync(dir)) continue;

    const files = fs.readdirSync(dir).filter(fn => /\.(png|jpg|jpeg|webp)$/i.test(fn)).sort();
    const colorFiles = files.filter(fn => fn.includes('_color'));
    const depthFiles = files.filter(fn => fn.includes('_depth'));
    const prefix = path.isAbsolute(f) ? '/file' + dir : '/images/' + f;
    
    // Find original BW image — look for the source image (no _color, _depth, _3d, _comparison suffix)
    // The original is usually the input file name without pipeline suffixes
    const baseName = path.basename(f);
    let originalPath = null;
    // Check common locations for the original
    const possibleOriginals = [
      path.join(dir, baseName + '.jpg'),
      path.join(dir, baseName + '.png'),
      // Check Downloads folder
      path.join(os.homedir(), 'Downloads', baseName + '.jpg'),
      path.join(os.homedir(), 'Downloads', baseName + '.jpeg'),
      path.join(os.homedir(), 'Downloads', baseName + '.png'),
    ];
    for (const p of possibleOriginals) {
      if (fs.existsSync(p)) { originalPath = p; break; }
    }

    if (colorFiles.length > 0) {
      for (const cf of colorFiles) {
        const base = cf.replace(/_color\.\w+$/, '');
        const df = depthFiles.find(d => d.startsWith(base + '_depth'));
        allPages.push({
          name: base,
          color: prefix + '/' + cf,
          depth: df ? prefix + '/' + df : null,
          original: originalPath ? '/file' + originalPath : null
        });
      }
    } else {
      // No _color convention — treat non-depth/non-comparison as pages
      const pageFiles = files.filter(fn => !fn.includes('_depth') && !fn.includes('_comparison') && !fn.includes('_3d'));
      for (const fn of pageFiles) {
        allPages.push({
          name: fn.replace(/\.\w+$/, ''),
          color: prefix + '/' + fn,
          depth: null,
          original: originalPath ? '/file' + originalPath : null
        });
      }
    }
  }
  
  res.json({ pages: allPages });
});

app.listen(PORT, () => {
  console.log(`🌌 Manga 3D Comic Reader running at http://localhost:${PORT}`);
});
