const { readFileSync } = require('fs');
const { loadPyodide } = require('/opt/pyodide/pyodide/pyodide.js');

async function main() {
    const data = JSON.parse(readFileSync('/json', 'utf8'));
    const sandboxRunCode = readFileSync('/sandbox_run.py', 'utf8');

    const pyodide = await loadPyodide({
        indexURL: '/opt/pyodide/pyodide'
    });

    const pythonCode = `
${sandboxRunCode}

data = ${JSON.stringify(data)}
result = run(data)

from json import dumps
print(dumps(result))
`;

    let stdout = [];
    let stderr = [];

    pyodide.setStdout({
        batched: (text) => {
            stdout.push(text);
        }
    });

    pyodide.setStderr({
        batched: (text) => {
            stderr.push(text);
        }
    });

    try {
        await pyodide.runPythonAsync(pythonCode);
        if (stdout.length > 0) console.log(stdout.join(''));
        if (stderr.length > 0) console.error(stderr.join(''));
    } catch (error) {
        if (stderr.length > 0) console.error(stderr.join(''));
        console.error(JSON.stringify({ error: error.message }));
    }
}

main();
