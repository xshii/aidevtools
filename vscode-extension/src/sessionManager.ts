import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { Artifact } from './types';

/**
 * Manages session folders and artifact files
 */
export class SessionManager {
    private sessionId: string;
    private sessionPath: string | null = null;

    constructor() {
        this.sessionId = this.generateSessionId();
    }

    /**
     * Generate session ID with datetime prefix
     */
    private generateSessionId(): string {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        const hour = String(now.getHours()).padStart(2, '0');
        const minute = String(now.getMinutes()).padStart(2, '0');
        const second = String(now.getSeconds()).padStart(2, '0');
        const random = Math.random().toString(36).substring(2, 6);

        return `${year}${month}${day}_${hour}${minute}${second}_${random}`;
    }

    /**
     * Get or create session folder in workspace tmp
     */
    getSessionPath(): string | null {
        if (this.sessionPath) {
            return this.sessionPath;
        }

        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders || workspaceFolders.length === 0) {
            return null;
        }

        const workspaceRoot = workspaceFolders[0].uri.fsPath;
        const tmpDir = path.join(workspaceRoot, 'tmp', 'prettycli');
        const sessionDir = path.join(tmpDir, this.sessionId);

        // Create directories if not exist
        if (!fs.existsSync(sessionDir)) {
            fs.mkdirSync(sessionDir, { recursive: true });
        }

        this.sessionPath = sessionDir;
        return sessionDir;
    }

    /**
     * Generate filename with datetime prefix
     */
    generateFilename(type: string, extension: string): string {
        const now = new Date();
        const hour = String(now.getHours()).padStart(2, '0');
        const minute = String(now.getMinutes()).padStart(2, '0');
        const second = String(now.getSeconds()).padStart(2, '0');
        const ms = String(now.getMilliseconds()).padStart(3, '0');
        const random = Math.random().toString(36).substring(2, 6);

        return `${hour}${minute}${second}_${ms}_${type}_${random}.${extension}`;
    }

    /**
     * Save artifact to session folder
     */
    saveArtifact(artifact: Artifact, content: string): string | null {
        const sessionPath = this.getSessionPath();
        if (!sessionPath) {
            return null;
        }

        const extension = this.getExtension(artifact.type);
        const filename = this.generateFilename(artifact.type, extension);
        const filePath = path.join(sessionPath, filename);

        fs.writeFileSync(filePath, content, 'utf-8');

        return filePath;
    }

    /**
     * Save binary artifact (like images)
     */
    saveBinaryArtifact(artifact: Artifact, data: Buffer, extension: string): string | null {
        const sessionPath = this.getSessionPath();
        if (!sessionPath) {
            return null;
        }

        const filename = this.generateFilename(artifact.type, extension);
        const filePath = path.join(sessionPath, filename);

        fs.writeFileSync(filePath, data);

        return filePath;
    }

    /**
     * Get file extension for artifact type
     */
    private getExtension(type: string): string {
        const extensions: Record<string, string> = {
            'chart': 'html',
            'table': 'html',
            'file': 'txt',
            'diff': 'html',
            'web': 'html',
            'markdown': 'md',
            'json': 'json',
            'image': 'png',
            'csv': 'csv',
            'xlsx': 'xlsx',
        };
        return extensions[type] || 'txt';
    }

    /**
     * List all artifacts in current session
     */
    listArtifacts(): string[] {
        const sessionPath = this.getSessionPath();
        if (!sessionPath || !fs.existsSync(sessionPath)) {
            return [];
        }

        return fs.readdirSync(sessionPath)
            .filter(f => !f.startsWith('.'))
            .sort()
            .reverse(); // Newest first
    }

    /**
     * List all sessions
     */
    listSessions(): { id: string; path: string; files: number }[] {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders || workspaceFolders.length === 0) {
            return [];
        }

        const tmpDir = path.join(workspaceFolders[0].uri.fsPath, 'tmp', 'prettycli');
        if (!fs.existsSync(tmpDir)) {
            return [];
        }

        return fs.readdirSync(tmpDir)
            .filter(f => fs.statSync(path.join(tmpDir, f)).isDirectory())
            .sort()
            .reverse()
            .map(id => {
                const sessionDir = path.join(tmpDir, id);
                const files = fs.readdirSync(sessionDir).length;
                return { id, path: sessionDir, files };
            });
    }

    /**
     * Clean up old sessions (keep last N)
     */
    cleanupOldSessions(keepCount: number = 10): number {
        const sessions = this.listSessions();
        let deletedCount = 0;

        if (sessions.length > keepCount) {
            const toDelete = sessions.slice(keepCount);
            for (const session of toDelete) {
                if (session.id !== this.sessionId) {
                    fs.rmSync(session.path, { recursive: true, force: true });
                    deletedCount++;
                }
            }
        }

        return deletedCount;
    }

    /**
     * Get current session ID
     */
    getSessionId(): string {
        return this.sessionId;
    }

    /**
     * Open session folder in explorer
     */
    async openSessionFolder(): Promise<void> {
        const sessionPath = this.getSessionPath();
        if (sessionPath) {
            const uri = vscode.Uri.file(sessionPath);
            await vscode.commands.executeCommand('revealFileInOS', uri);
        }
    }

    /**
     * Open artifact file in editor
     */
    async openArtifactFile(filename: string): Promise<void> {
        const sessionPath = this.getSessionPath();
        if (!sessionPath) {
            return;
        }

        const filePath = path.join(sessionPath, filename);
        if (fs.existsSync(filePath)) {
            const uri = vscode.Uri.file(filePath);
            await vscode.window.showTextDocument(uri);
        }
    }
}

// Global session manager
export const sessionManager = new SessionManager();
