const { readFileSync } = require('fs');
const { loadPyodide } = require('/opt/pyodide/pyodide/pyodide.js');

async function main() {
    const data = JSON.parse(readFileSync('/json', 'utf8'));

    const pyodide = await loadPyodide({
        indexURL: '/opt/pyodide/pyodide'
    });

    const pythonCode = `
from math import *
from json import dumps

${data.fn}

try:
    if "blackbox" not in globals():
        print(dumps({"error": "function \`blackbox\` not defined"}))
    else:
        result = blackbox(**${JSON.stringify(data.kwargs)})
        print(dumps({"result": result}))
except Exception as e:
    print(dumps({"error": str(e)}))
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
