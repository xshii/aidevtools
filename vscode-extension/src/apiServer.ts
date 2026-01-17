import * as vscode from 'vscode';
import { WebSocketServer, WebSocket } from 'ws';
import { ApiMessage, ApiResponse } from './types';
import { PanelManager } from './panelManager';

/**
 * WebSocket API server for CLI communication
 */
export class ApiServer {
    private server: WebSocketServer | null = null;
    private panelManager: PanelManager;
    private outputChannel: vscode.OutputChannel;

    constructor(panelManager: PanelManager) {
        this.panelManager = panelManager;
        this.outputChannel = vscode.window.createOutputChannel('prettycli');
    }

    /**
     * Start the API server
     */
    start(port: number): void {
        if (this.server) {
            this.stop();
        }

        try {
            this.server = new WebSocketServer({ port });

            this.server.on('connection', (ws) => {
                this.log(`Client connected`);

                ws.on('message', (data) => {
                    this.handleMessage(ws, data.toString());
                });

                ws.on('close', () => {
                    this.log(`Client disconnected`);
                });

                ws.on('error', (error) => {
                    this.log(`WebSocket error: ${error.message}`);
                });
            });

            this.server.on('error', (error) => {
                this.log(`Server error: ${error.message}`);
                vscode.window.showErrorMessage(`prettycli: Server error - ${error.message}`);
            });

            this.log(`API server started on port ${port}`);
        } catch (error) {
            this.log(`Failed to start server: ${error}`);
            throw error;
        }
    }

    /**
     * Stop the API server
     */
    stop(): void {
        if (this.server) {
            this.server.close();
            this.server = null;
            this.log('API server stopped');
        }
    }

    /**
     * Handle incoming message
     */
    private handleMessage(ws: WebSocket, data: string): void {
        let message: ApiMessage;

        try {
            message = JSON.parse(data);
        } catch {
            this.sendResponse(ws, {
                id: 'unknown',
                success: false,
                error: 'Invalid JSON',
            });
            return;
        }

        this.log(`Received: ${message.action} (${message.id})`);

        try {
            const response = this.processMessage(message);
            this.sendResponse(ws, response);
        } catch (error) {
            this.sendResponse(ws, {
                id: message.id,
                success: false,
                error: error instanceof Error ? error.message : String(error),
            });
        }
    }

    /**
     * Process API message
     */
    private processMessage(message: ApiMessage): ApiResponse {
        switch (message.action) {
            case 'render':
                if (!message.artifact) {
                    return {
                        id: message.id,
                        success: false,
                        error: 'Missing artifact',
                    };
                }
                const result = this.panelManager.show(message.artifact, message.panelId);
                return {
                    id: message.id,
                    success: true,
                    data: { panelId: result.panelId, filePath: result.filePath },
                };

            case 'update':
                if (!message.panelId || !message.artifact) {
                    return {
                        id: message.id,
                        success: false,
                        error: 'Missing panelId or artifact',
                    };
                }
                const updateResult = this.panelManager.update(message.panelId, message.artifact);
                return {
                    id: message.id,
                    success: updateResult.success,
                    data: updateResult.success ? { filePath: updateResult.filePath } : undefined,
                    error: updateResult.success ? undefined : 'Panel not found',
                };

            case 'close':
                if (!message.panelId) {
                    return {
                        id: message.id,
                        success: false,
                        error: 'Missing panelId',
                    };
                }
                const closed = this.panelManager.close(message.panelId);
                return {
                    id: message.id,
                    success: closed,
                    error: closed ? undefined : 'Panel not found',
                };

            case 'list':
                return {
                    id: message.id,
                    success: true,
                    data: { panels: this.panelManager.list() },
                };

            case 'open':
                if (!message.artifact?.path) {
                    return {
                        id: message.id,
                        success: false,
                        error: 'Missing path',
                    };
                }
                // Open file with system default application
                vscode.env.openExternal(vscode.Uri.file(message.artifact.path));
                return {
                    id: message.id,
                    success: true,
                };

            case 'ping':
                return {
                    id: message.id,
                    success: true,
                };

            default:
                return {
                    id: message.id,
                    success: false,
                    error: `Unknown action: ${message.action}`,
                };
        }
    }

    /**
     * Send response to client
     */
    private sendResponse(ws: WebSocket, response: ApiResponse): void {
        ws.send(JSON.stringify(response));
        this.log(`Response: ${response.success ? 'success' : 'error'} (${response.id})`);
    }

    /**
     * Log message to output channel
     */
    private log(message: string): void {
        const timestamp = new Date().toISOString();
        this.outputChannel.appendLine(`[${timestamp}] ${message}`);
    }

    /**
     * Get server status
     */
    isRunning(): boolean {
        return this.server !== null;
    }

    /**
     * Dispose resources
     */
    dispose(): void {
        this.stop();
        this.outputChannel.dispose();
    }
}
