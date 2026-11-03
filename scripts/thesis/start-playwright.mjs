import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const cli = path.join(root, 'tools', 'thesis-agent', 'node_modules', '@playwright', 'mcp', 'cli.js');
const child = spawn(process.execPath, [cli, '--browser=chrome',
  '--user-data-dir', path.join(root, '.local', 'overleaf-profile'),
  '--output-dir', path.join(root, '.local', 'mcp-output')], {cwd: root, stdio: 'inherit'});
child.on('error', e => { console.error('Run npm ci --prefix tools/thesis-agent first:', e.message); process.exit(1); });
child.on('exit', code => process.exit(code ?? 1));
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
