import { ChartRenderer } from '../src/renderers/chart';
import { TableRenderer } from '../src/renderers/table';
import { FileRenderer } from '../src/renderers/file';
import { DiffRenderer } from '../src/renderers/diff';
import { WebRenderer } from '../src/renderers/web';
import { MarkdownRenderer } from '../src/renderers/markdown';
import { JsonRenderer } from '../src/renderers/json';
import { ImageRenderer } from '../src/renderers/image';
import { escapeHtml, getBaseHtml } from '../src/renderers/base';

describe('escapeHtml', () => {
  it('should escape special characters', () => {
    expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
    expect(escapeHtml('"test"')).toBe('&quot;test&quot;');
    expect(escapeHtml("'test'")).toBe('&#039;test&#039;');
    expect(escapeHtml('a & b')).toBe('a &amp; b');
  });

  it('should handle normal text', () => {
    expect(escapeHtml('hello world')).toBe('hello world');
  });
});

describe('getBaseHtml', () => {
  it('should wrap content in HTML structure', () => {
    const html = getBaseHtml('<p>test</p>', 'Test Title');

    expect(html).toContain('<!DOCTYPE html>');
    expect(html).toContain('<title>Test Title</title>');
    expect(html).toContain('<p>test</p>');
  });
});

describe('ChartRenderer', () => {
  const renderer = new ChartRenderer();

  it('should have correct type', () => {
    expect(renderer.type).toBe('chart');
  });

  it('should render bar chart', () => {
    const artifact = {
      type: 'chart' as const,
      title: 'Test Chart',
      data: {
        chartType: 'bar' as const,
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
      type: 'chart' as const,
      data: {
        chartType: 'line' as const,
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
      type: 'chart' as const,
      data: {
        chartType: 'pie' as const,
        labels: ['A', 'B'],
        datasets: [{ label: 'Pie', data: [30, 70] }],
      },
    };

    const html = renderer.render(artifact);

    expect(html).toContain('path');
  });
});

describe('TableRenderer', () => {
  const renderer = new TableRenderer();

  it('should have correct type', () => {
    expect(renderer.type).toBe('table');
  });

  it('should render table with data', () => {
    const artifact = {
      type: 'table' as const,
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
  const renderer = new FileRenderer();

  it('should have correct type', () => {
    expect(renderer.type).toBe('file');
  });

  it('should render file content', () => {
    const artifact = {
      type: 'file' as const,
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
      type: 'file' as const,
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
  const renderer = new DiffRenderer();

  it('should have correct type', () => {
    expect(renderer.type).toBe('diff');
  });

  it('should render diff', () => {
    const artifact = {
      type: 'diff' as const,
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
  const renderer = new WebRenderer();

  it('should have correct type', () => {
    expect(renderer.type).toBe('web');
  });

  it('should render HTML content', () => {
    const artifact = {
      type: 'web' as const,
      data: {
        html: '<h1>Hello</h1>',
      },
    };

    const html = renderer.render(artifact);

    expect(html).toContain('<h1>Hello</h1>');
  });

  it('should render URL', () => {
    const artifact = {
      type: 'web' as const,
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
      type: 'web' as const,
      data: {},
    };

    const html = renderer.render(artifact);

    expect(html).toContain('No content');
  });
});

describe('MarkdownRenderer', () => {
  const renderer = new MarkdownRenderer();

  it('should have correct type', () => {
    expect(renderer.type).toBe('markdown');
  });

  it('should render markdown', () => {
    const artifact = {
      type: 'markdown' as const,
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
      type: 'markdown' as const,
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
  const renderer = new JsonRenderer();

  it('should have correct type', () => {
    expect(renderer.type).toBe('json');
  });

  it('should render JSON object', () => {
    const artifact = {
      type: 'json' as const,
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
      type: 'json' as const,
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
      type: 'json' as const,
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
  const renderer = new ImageRenderer();

  it('should have correct type', () => {
    expect(renderer.type).toBe('image');
  });

  it('should render image', () => {
    const artifact = {
      type: 'image' as const,
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
