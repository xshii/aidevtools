"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const chart_1 = require("../src/renderers/chart");
const table_1 = require("../src/renderers/table");
const file_1 = require("../src/renderers/file");
const diff_1 = require("../src/renderers/diff");
const web_1 = require("../src/renderers/web");
const markdown_1 = require("../src/renderers/markdown");
const json_1 = require("../src/renderers/json");
const image_1 = require("../src/renderers/image");
const base_1 = require("../src/renderers/base");
describe('escapeHtml', () => {
    it('should escape special characters', () => {
        expect((0, base_1.escapeHtml)('<script>')).toBe('&lt;script&gt;');
        expect((0, base_1.escapeHtml)('"test"')).toBe('&quot;test&quot;');
        expect((0, base_1.escapeHtml)("'test'")).toBe('&#039;test&#039;');
        expect((0, base_1.escapeHtml)('a & b')).toBe('a &amp; b');
    });
    it('should handle normal text', () => {
        expect((0, base_1.escapeHtml)('hello world')).toBe('hello world');
    });
});
describe('getBaseHtml', () => {
    it('should wrap content in HTML structure', () => {
        const html = (0, base_1.getBaseHtml)('<p>test</p>', 'Test Title');
        expect(html).toContain('<!DOCTYPE html>');
        expect(html).toContain('<title>Test Title</title>');
        expect(html).toContain('<p>test</p>');
    });
});
describe('ChartRenderer', () => {
    const renderer = new chart_1.ChartRenderer();
    it('should have correct type', () => {
        expect(renderer.type).toBe('chart');
    });
    it('should render bar chart', () => {
        const artifact = {
            type: 'chart',
            title: 'Test Chart',
            data: {
                chartType: 'bar',
                labels: ['A', 'B'],
                datasets: [{ label: 'Test', data: [10, 20] }],
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('svg');
        expect(html).toContain('Test Chart');
        expect(html).toContain('rect');
    });
    it('should render line chart', () => {
        const artifact = {
            type: 'chart',
            data: {
                chartType: 'line',
                labels: ['A', 'B', 'C'],
                datasets: [{ label: 'Line', data: [1, 2, 3] }],
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('polyline');
        expect(html).toContain('circle');
    });
    it('should render pie chart', () => {
        const artifact = {
            type: 'chart',
            data: {
                chartType: 'pie',
                labels: ['A', 'B'],
                datasets: [{ label: 'Pie', data: [30, 70] }],
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('path');
    });
});
describe('TableRenderer', () => {
    const renderer = new table_1.TableRenderer();
    it('should have correct type', () => {
        expect(renderer.type).toBe('table');
    });
    it('should render table with data', () => {
        const artifact = {
            type: 'table',
            title: 'Test Table',
            data: {
                columns: ['Name', 'Value'],
                rows: [['A', 1], ['B', 2]],
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('<table>');
        expect(html).toContain('<th>Name</th>');
        expect(html).toContain('<td>A</td>');
        expect(html).toContain('2 rows');
    });
});
describe('FileRenderer', () => {
    const renderer = new file_1.FileRenderer();
    it('should have correct type', () => {
        expect(renderer.type).toBe('file');
    });
    it('should render file content', () => {
        const artifact = {
            type: 'file',
            data: {
                path: '/test/file.py',
                content: 'print("hello")\nprint("world")',
                language: 'python',
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('file.py');
        expect(html).toContain('python');
        expect(html).toContain('print');
    });
    it('should handle line range', () => {
        const artifact = {
            type: 'file',
            data: {
                path: '/test/file.txt',
                content: 'line1\nline2\nline3\nline4',
                startLine: 2,
                endLine: 3,
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('Lines 2-3');
    });
});
describe('DiffRenderer', () => {
    const renderer = new diff_1.DiffRenderer();
    it('should have correct type', () => {
        expect(renderer.type).toBe('diff');
    });
    it('should render diff', () => {
        const artifact = {
            type: 'diff',
            data: {
                original: 'line1\nline2',
                modified: 'line1\nline2\nline3',
                originalPath: 'old.txt',
                modifiedPath: 'new.txt',
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('old.txt');
        expect(html).toContain('new.txt');
        expect(html).toContain('added');
    });
});
describe('WebRenderer', () => {
    const renderer = new web_1.WebRenderer();
    it('should have correct type', () => {
        expect(renderer.type).toBe('web');
    });
    it('should render HTML content', () => {
        const artifact = {
            type: 'web',
            data: {
                html: '<h1>Hello</h1>',
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('<h1>Hello</h1>');
    });
    it('should render URL', () => {
        const artifact = {
            type: 'web',
            data: {
                url: 'https://example.com',
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('iframe');
        expect(html).toContain('https://example.com');
    });
    it('should handle empty content', () => {
        const artifact = {
            type: 'web',
            data: {},
        };
        const html = renderer.render(artifact);
        expect(html).toContain('No content');
    });
});
describe('MarkdownRenderer', () => {
    const renderer = new markdown_1.MarkdownRenderer();
    it('should have correct type', () => {
        expect(renderer.type).toBe('markdown');
    });
    it('should render markdown', () => {
        const artifact = {
            type: 'markdown',
            data: {
                content: '# Hello\n**bold** and *italic*',
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('<h1>');
        expect(html).toContain('<strong>');
        expect(html).toContain('<em>');
    });
    it('should render code blocks', () => {
        const artifact = {
            type: 'markdown',
            data: {
                content: '```python\nprint("hi")\n```',
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('<pre>');
        expect(html).toContain('<code');
    });
});
describe('JsonRenderer', () => {
    const renderer = new json_1.JsonRenderer();
    it('should have correct type', () => {
        expect(renderer.type).toBe('json');
    });
    it('should render JSON object', () => {
        const artifact = {
            type: 'json',
            data: {
                content: { name: 'test', value: 123 },
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('json-key');
        expect(html).toContain('json-string');
        expect(html).toContain('json-number');
    });
    it('should render arrays', () => {
        const artifact = {
            type: 'json',
            data: {
                content: [1, 2, 3],
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('[');
        expect(html).toContain(']');
    });
    it('should handle null and boolean', () => {
        const artifact = {
            type: 'json',
            data: {
                content: { isNull: null, isTrue: true, isFalse: false },
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('json-null');
        expect(html).toContain('json-boolean');
    });
});
describe('ImageRenderer', () => {
    const renderer = new image_1.ImageRenderer();
    it('should have correct type', () => {
        expect(renderer.type).toBe('image');
    });
    it('should render image', () => {
        const artifact = {
            type: 'image',
            title: 'Test Image',
            data: {
                src: 'data:image/png;base64,abc123',
                alt: 'Test alt text',
                width: 100,
                height: 50,
            },
        };
        const html = renderer.render(artifact);
        expect(html).toContain('<img');
        expect(html).toContain('data:image/png;base64,abc123');
        expect(html).toContain('Test alt text');
        expect(html).toContain('width: 100px');
        expect(html).toContain('height: 50px');
    });
});
//# sourceMappingURL=renderers.test.js.map