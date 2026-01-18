/**
 * Mock VS Code API for testing
 */

export const window = {
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

export const workspace = {
  workspaceFolders: [
    {
      uri: {
        fsPath: '/mock/workspace',
      },
    },
  ],
  getConfiguration: jest.fn(() => ({
    get: jest.fn((key: string, defaultValue: any) => defaultValue),
  })),
  onDidChangeConfiguration: jest.fn(),
};

export const commands = {
  registerCommand: jest.fn(),
  executeCommand: jest.fn(),
};

export const Uri = {
  file: jest.fn((path: string) => ({ fsPath: path })),
};

export enum ViewColumn {
  Beside = 2,
}

export class EventEmitter {
  private listeners: Function[] = [];

  event = (listener: Function) => {
    this.listeners.push(listener);
    return { dispose: () => {} };
  };

  fire(data: any) {
    this.listeners.forEach(l => l(data));
  }
}
