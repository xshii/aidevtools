/**
 * Base HTML template for webview
 */
export function getBaseHtml(content: string, title: string = 'Artifact'): string {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${escapeHtml(title)}</title>
    <style>
        :root {
            --bg-color: var(--vscode-editor-background);
            --fg-color: var(--vscode-editor-foreground);
            --border-color: var(--vscode-panel-border);
            --header-bg: var(--vscode-sideBarSectionHeader-background);
            --link-color: var(--vscode-textLink-foreground);
            --success-color: #4caf50;
            --error-color: #f44336;
            --warning-color: #ff9800;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--fg-color);
            background-color: var(--bg-color);
            margin: 0;
            padding: 16px;
            line-height: 1.5;
        }

        h1, h2, h3 {
            margin-top: 0;
            color: var(--fg-color);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 8px 0;
        }

        th, td {
            padding: 8px 12px;
            text-align: left;
            border: 1px solid var(--border-color);
        }

        th {
            background-color: var(--header-bg);
            font-weight: 600;
        }

        tr:hover {
            background-color: var(--vscode-list-hoverBackground);
        }

        pre, code {
            font-family: var(--vscode-editor-font-family);
            font-size: var(--vscode-editor-font-size);
            background-color: var(--vscode-textCodeBlock-background);
            border-radius: 4px;
        }

        pre {
            padding: 12px;
            overflow-x: auto;
            margin: 8px 0;
        }

        code {
            padding: 2px 6px;
        }

        .chart-container {
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
        }

        .diff-container {
            display: flex;
            gap: 16px;
        }

        .diff-panel {
            flex: 1;
            overflow-x: auto;
        }

        .diff-header {
            padding: 8px;
            background-color: var(--header-bg);
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
        }

        .line-added {
            background-color: rgba(76, 175, 80, 0.2);
        }

        .line-removed {
            background-color: rgba(244, 67, 54, 0.2);
        }

        .file-content {
            border: 1px solid var(--border-color);
            border-radius: 4px;
            overflow: hidden;
        }

        .file-header {
            padding: 8px 12px;
            background-color: var(--header-bg);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .file-path {
            font-family: var(--vscode-editor-font-family);
            color: var(--link-color);
        }

        .json-key {
            color: var(--vscode-symbolIcon-propertyForeground, #9cdcfe);
        }

        .json-string {
            color: var(--vscode-symbolIcon-stringForeground, #ce9178);
        }

        .json-number {
            color: var(--vscode-symbolIcon-numberForeground, #b5cea8);
        }

        .json-boolean {
            color: var(--vscode-symbolIcon-booleanForeground, #569cd6);
        }

        .json-null {
            color: var(--vscode-symbolIcon-nullForeground, #569cd6);
        }

        .collapsible {
            cursor: pointer;
        }

        .collapsible::before {
            content: '▼ ';
            font-size: 0.8em;
        }

        .collapsed::before {
            content: '▶ ';
        }

        .collapsed + .collapsible-content {
            display: none;
        }

        iframe {
            border: none;
            width: 100%;
            min-height: 400px;
        }
    </style>
</head>
<body>
    ${content}
</body>
</html>
    `.trim();
}

/**
 * Escape HTML special characters
 */
export function escapeHtml(text: string): string {
    const map: Record<string, string> = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, char => map[char]);
}
