import * as vscode from 'vscode';
import { PanelManager } from './panelManager';
import { ApiServer } from './apiServer';
import { registerBuiltinRenderers } from './renderers';
import { pluginManager } from './pluginManager';

let panelManager: PanelManager;
let apiServer: ApiServer;

/**
 * Extension activation
 */
export function activate(context: vscode.ExtensionContext) {
    // Register built-in renderers
    registerBuiltinRenderers();

    // Initialize panel manager
    panelManager = new PanelManager(context);

    // Initialize and start API server
    apiServer = new ApiServer(panelManager);

    const config = vscode.workspace.getConfiguration('prettycli');
    const port = config.get<number>('serverPort', 19960);

    try {
        apiServer.start(port);
    } catch (error) {
        vscode.window.showErrorMessage(`prettycli: Failed to start API server on port ${port}`);
    }

    // Register commands
    const showArtifactCommand = vscode.commands.registerCommand('prettycli.showArtifact', async () => {
        const type = await vscode.window.showQuickPick(
            pluginManager.listTypes(),
            { placeHolder: 'Select artifact type' }
        );

        if (!type) {
            return;
        }

        // Demo artifact
        const demoArtifact = getDemoArtifact(type);
        if (demoArtifact) {
            panelManager.show(demoArtifact);
        }
    });

    // Watch for configuration changes
    const configWatcher = vscode.workspace.onDidChangeConfiguration((e) => {
        if (e.affectsConfiguration('prettycli.serverPort')) {
            const newPort = vscode.workspace.getConfiguration('prettycli').get<number>('serverPort', 19960);
            apiServer.stop();
            try {
                apiServer.start(newPort);
                vscode.window.showInformationMessage(`prettycli: Server restarted on port ${newPort}`);
            } catch (error) {
                vscode.window.showErrorMessage(`prettycli: Failed to restart server on port ${newPort}`);
            }
        }
    });

    context.subscriptions.push(
        showArtifactCommand,
        configWatcher,
        { dispose: () => apiServer.dispose() },
        { dispose: () => panelManager.dispose() }
    );

    // Export API for other extensions
    return {
        /**
         * Show an artifact
         */
        showArtifact: (artifact: unknown, panelId?: string) => {
            return panelManager.show(artifact as any, panelId);
        },

        /**
         * Update an artifact panel
         */
        updateArtifact: (panelId: string, artifact: unknown) => {
            return panelManager.update(panelId, artifact as any);
        },

        /**
         * Close a panel
         */
        closePanel: (panelId: string) => {
            return panelManager.close(panelId);
        },

        /**
         * List open panels
         */
        listPanels: () => {
            return panelManager.list();
        },

        /**
         * Register a custom renderer
         */
        registerRenderer: (renderer: unknown) => {
            pluginManager.register(renderer as any);
        },

        /**
         * Get server port
         */
        getServerPort: () => {
            return vscode.workspace.getConfiguration('prettycli').get<number>('serverPort', 19960);
        },
    };
}

/**
 * Extension deactivation
 */
export function deactivate() {
    if (apiServer) {
        apiServer.dispose();
    }
    if (panelManager) {
        panelManager.dispose();
    }
}

/**
 * Get demo artifact for testing
 */
function getDemoArtifact(type: string): any {
    switch (type) {
        case 'chart':
            return {
                type: 'chart',
                title: 'Demo Chart',
                data: {
                    chartType: 'bar',
                    labels: ['Jan', 'Feb', 'Mar', 'Apr'],
                    datasets: [
                        { label: 'Sales', data: [100, 150, 120, 180] },
                        { label: 'Revenue', data: [80, 120, 100, 150] },
                    ],
                },
            };

        case 'table':
            return {
                type: 'table',
                title: 'Demo Table',
                data: {
                    columns: ['Name', 'Age', 'City'],
                    rows: [
                        ['Alice', 25, 'New York'],
                        ['Bob', 30, 'Los Angeles'],
                        ['Charlie', 35, 'Chicago'],
                    ],
                },
            };

        case 'file':
            return {
                type: 'file',
                title: 'Demo File',
                data: {
                    path: 'src/example.py',
                    language: 'python',
                    content: `def hello_world():
    """Print hello world."""
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()`,
                },
            };

        case 'diff':
            return {
                type: 'diff',
                title: 'Demo Diff',
                data: {
                    original: `function greet(name) {
    console.log("Hello, " + name);
}`,
                    modified: `function greet(name: string): void {
    console.log(\`Hello, \${name}!\`);
}

export { greet };`,
                    originalPath: 'greet.js',
                    modifiedPath: 'greet.ts',
                    language: 'typescript',
                },
            };

        case 'web':
            return {
                type: 'web',
                title: 'Demo Web',
                data: {
                    html: '<h2>Hello from prettycli!</h2><p>This is rendered HTML content.</p>',
                },
            };

        case 'markdown':
            return {
                type: 'markdown',
                title: 'Demo Markdown',
                data: {
                    content: `# Hello World

This is a **markdown** document with *formatting*.

## Features

* Item one
* Item two
* Item three

\`\`\`python
print("Hello!")
\`\`\`
`,
                },
            };

        case 'json':
            return {
                type: 'json',
                title: 'Demo JSON',
                data: {
                    content: {
                        name: 'prettycli',
                        version: '0.1.0',
                        features: ['chart', 'table', 'diff'],
                        config: {
                            port: 19960,
                            enabled: true,
                        },
                    },
                },
            };

        case 'image':
            return {
                type: 'image',
                title: 'Demo Image',
                data: {
                    // Simple 1x1 red pixel as placeholder
                    src: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
                    alt: 'Demo image placeholder',
                },
            };

        default:
            return null;
    }
}
