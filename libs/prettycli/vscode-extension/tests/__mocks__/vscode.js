"use strict";
/**
 * Mock VS Code API for testing
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.EventEmitter = exports.ViewColumn = exports.Uri = exports.commands = exports.workspace = exports.window = void 0;
exports.window = {
    createWebviewPanel: jest.fn(() => ({
        webview: {
            html: '',
        },
        title: '',
        onDidDispose: jest.fn(),
        dispose: jest.fn(),
    })),
    showQuickPick: jest.fn(),
    showInformationMessage: jest.fn(),
    showErrorMessage: jest.fn(),
    createOutputChannel: jest.fn(() => ({
        appendLine: jest.fn(),
        dispose: jest.fn(),
    })),
};
exports.workspace = {
    workspaceFolders: [
        {
            uri: {
                fsPath: '/mock/workspace',
            },
        },
    ],
    getConfiguration: jest.fn(() => ({
        get: jest.fn((key, defaultValue) => defaultValue),
    })),
    onDidChangeConfiguration: jest.fn(),
};
exports.commands = {
    registerCommand: jest.fn(),
    executeCommand: jest.fn(),
};
exports.Uri = {
    file: jest.fn((path) => ({ fsPath: path })),
};
var ViewColumn;
(function (ViewColumn) {
    ViewColumn[ViewColumn["Beside"] = 2] = "Beside";
})(ViewColumn || (exports.ViewColumn = ViewColumn = {}));
class EventEmitter {
    constructor() {
        this.listeners = [];
        this.event = (listener) => {
            this.listeners.push(listener);
            return { dispose: () => { } };
        };
    }
    fire(data) {
        this.listeners.forEach(l => l(data));
    }
}
exports.EventEmitter = EventEmitter;
//# sourceMappingURL=vscode.js.map