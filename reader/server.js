const express = require('express');
const path = require('path');
const fs = require('fs');
const os = require('os');
const multer = require('multer');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3002;
const ROOT_DIR = path.resolve(__dirname, '..');
const OUTPUT_DIR = path.resolve(ROOT_DIR, 'output');
const UPLOADS_DIR = path.resolve(ROOT_DIR, 'uploads');
const PIPELINE_SCRIPT = path.resolve(ROOT_DIR, 'pipeline', 'manga_pipeline.py');

// Ensure dirs exist
[OUTPUT_DIR, UPLOADS_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));

// Track active transform jobs: jobId -> {status, progress, outputDir, pages, error}
const jobs = new Map();

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve images from the manga output directory
app.use('/images', express.static(OUTPUT_DIR));

// Serve uploaded files
app.use('/uploads', express.static(UPLOADS_DIR));

// Serve arbitrary files by absolute path
app.get('/file/*', (req, res) => {
  const filePath = '/' + req.params[0];
  if (!fs.existsSync(filePath)) return res.status(404).send('Not found');
  res.sendFile(filePath);
});

// ---- File upload (multer) ----
const upload = multer({
  dest: UPLOADS_DIR,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
  fileFilter: (req, file, cb) => {
    if (/\.(png|jpg|jpeg|webp|bmp|pdf)$/i.test(file.originalname)) cb(null, true);
    else cb(new Error('Only image/PDF files allowed'));
  }
});

// API: upload images for transformation
app.post('/api/upload', upload.array('files', 50), (req, res) => {
  if (!req.files || !req.files.length) return res.status(400).json({ error: 'No files uploaded' });
  
  const jobId = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  const jobDir = path.join(UPLOADS_DIR, jobId);
  fs.mkdirSync(jobDir, { recursive: true });
  
  // Move uploaded files to job directory with original names
  const filePaths = [];
  for (const file of req.files) {
    const dest = path.join(jobDir, file.originalname);
    fs.renameSync(file.path, dest);
    filePaths.push(dest);
  }
  
  res.json({ jobId, files: filePaths.map(f => path.basename(f)) });
});

// API: start transform pipeline
app.post('/api/transform', (req, res) => {
  const { jobId, steps = 'all', device = 'auto' } = req.body;
  if (!jobId) return res.status(400).json({ error: 'jobId required' });
  
  const jobDir = path.join(UPLOADS_DIR, jobId);
  if (!fs.existsSync(jobDir)) return res.status(404).json({ error: 'Job not found' });
  
  const outputDir = path.join(OUTPUT_DIR, jobId);
  fs.mkdirSync(outputDir, { recursive: true });
  
  const job = { status: 'running', progress: [], outputDir, pages: [], error: null, startTime: Date.now() };
  jobs.set(jobId, job);
  
  // Find python (try conda manga3d env, then system python)
  const pythonCandidates = [
    path.join(os.homedir(), 'miniconda3', 'envs', 'manga3d', 'bin', 'python'),
    path.join(os.homedir(), 'anaconda3', 'envs', 'manga3d', 'bin', 'python'),
    path.join(os.homedir(), 'mambaforge', 'envs', 'manga3d', 'bin', 'python'),
    'python3', 'python'
  ];
  
  let pythonBin = 'python3';
  for (const p of pythonCandidates) {
    try {
      if (p.includes('/') && fs.existsSync(p)) { pythonBin = p; break; }
    } catch {}
  }
  
  const args = [PIPELINE_SCRIPT, jobDir, '-o', outputDir, '-s', steps, '--device', device, '--json-progress'];
  console.log(`🚀 Transform job ${jobId}: ${pythonBin} ${args.join(' ')}`);
  
  const proc = spawn(pythonBin, args, {
    cwd: ROOT_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' }
  });
  
  let stdoutBuf = '';
  proc.stdout.on('data', (data) => {
    stdoutBuf += data.toString();
    const lines = stdoutBuf.split('\n');
    stdoutBuf = lines.pop(); // keep incomplete line
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const msg = JSON.parse(trimmed);
        job.progress.push(msg);
      } catch {
        job.progress.push({ phase: 'log', msg: trimmed });
      }
    }
  });
  
  proc.stderr.on('data', (data) => {
    const text = data.toString().trim();
    if (text) job.progress.push({ phase: 'log', msg: text });
  });
  
  proc.on('close', (code) => {
    if (code === 0) {
      job.status = 'done';
      // Scan output dir for color/depth pairs
      try {
        const files = fs.readdirSync(outputDir).filter(f => /\.(png|jpg|jpeg|webp)$/i.test(f)).sort();
        const colorFiles = files.filter(f => f.includes('_color'));
        const depthFiles = files.filter(f => f.includes('_depth'));
        for (const cf of colorFiles) {
          const base = cf.replace(/_color\.\w+$/, '');
          const df = depthFiles.find(d => d.startsWith(base + '_depth'));
          job.pages.push({
            name: base,
            color: `/images/${jobId}/${cf}`,
            depth: df ? `/images/${jobId}/${df}` : null
          });
        }
      } catch {}
    } else {
      job.status = 'error';
      job.error = `Pipeline exited with code ${code}`;
    }
    job.endTime = Date.now();
    console.log(`${job.status === 'done' ? '✅' : '❌'} Job ${jobId} ${job.status} (${((job.endTime - job.startTime)/1000).toFixed(1)}s)`);
  });
  
  res.json({ jobId, status: 'running' });
});

// API: check transform status
app.get('/api/transform/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: 'Job not found' });
  res.json({
    status: job.status,
    progress: job.progress,
    pages: job.pages,
    error: job.error,
    elapsed: job.endTime ? job.endTime - job.startTime : Date.now() - job.startTime
  });
});

// API: list available output folders
app.get('/api/outputs', (req, res) => {
  try {
    const folders = fs.readdirSync(OUTPUT_DIR)
      .filter(f => fs.statSync(path.join(OUTPUT_DIR, f)).isDirectory())
      .map(f => {
        const files = fs.readdirSync(path.join(OUTPUT_DIR, f));
        const hasColor = files.some(fn => fn.includes('_color'));
        return { name: f, hasColor, fileCount: files.length };
      });
    res.json({ folders });
  } catch { res.json({ folders: [] }); }
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
    
    // Find original BW image
    const baseName = path.basename(f);
    let originalPath = null;
    const possibleOriginals = [
      path.join(dir, baseName + '.jpg'),
      path.join(dir, baseName + '.png'),
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
  console.log(`   Pipeline: ${PIPELINE_SCRIPT}`);
  console.log(`   Output: ${OUTPUT_DIR}`);
});
