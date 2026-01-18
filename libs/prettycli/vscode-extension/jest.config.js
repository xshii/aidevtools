/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  testMatch: ['**/*.test.ts'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/extension.ts',      // Requires VS Code runtime
    '!src/apiServer.ts',      // Requires VS Code + WebSocket
    '!src/panelManager.ts',   // Requires VS Code webview
    '!src/sessionManager.ts', // Requires VS Code workspace
    '!src/renderers/index.ts', // Re-exports only
  ],
  coverageDirectory: 'coverage',
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  moduleNameMapper: {
    '^vscode$': '<rootDir>/tests/__mocks__/vscode.ts',
  },
};
