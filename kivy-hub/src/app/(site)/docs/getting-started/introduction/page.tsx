'use client';

import { MarkdownRenderer } from '@/components/markdown/markdown-renderer';

const content = `# Introduction to Kivy

Welcome to Kivy's documentation. This guide will help you understand the core concepts
and get started with using Kivy to transform any surface into an interactive display.

## What is Kivy?

Kivy is a powerful framework that enables you to create interactive surfaces using
computer vision and projection mapping. It combines advanced AI techniques with
intuitive user interfaces to create immersive experiences.

## Key Features

- Real-time surface detection and tracking
- Advanced hand gesture recognition
- Seamless projection mapping
- Cross-platform compatibility
- Extensive API for customization

## Getting Started

To get started with Kivy, you'll need:

1. A camera with at least 720p resolution
2. A projector with minimum 1080p resolution
3. A computer with WebGL support
4. Node.js 16+ and npm/bun

## Code Example

Here's a simple example of initializing Kivy:

\`\`\`typescript
const kivy = new Kivy({
  camera: {
    width: 1280,
    height: 720
  },
  projector: {
    width: 1920,
    height: 1080
  }
});

// Start tracking
await kivy.startTracking();
\`\`\`

## Next Steps

- [Installation Guide](/docs/getting-started/installation)
- [Quick Start](/docs/getting-started/quick-start)
- [API Reference](/docs/api-reference)
`;

export default function IntroductionPage() {
  return <MarkdownRenderer content={content} />;
} 