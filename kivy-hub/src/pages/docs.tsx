import { useState } from 'react';
import {
  BookOpen,
  Code,
  Brain,
  Calculator,
  Settings,
  ChevronRight
} from 'lucide-react';

const sections = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    icon: Settings,
    content: (
      <div className='prose prose-indigo max-w-none'>
        <h2>Getting Started with Kivy</h2>
        <p>
          Welcome to Kivy's documentation. This guide will help you understand
          the core concepts and get started with using Kivy to transform any
          surface into an interactive display.
        </p>
        <h3>Prerequisites</h3>
        <ul>
          <li>Camera with at least 720p resolution</li>
          <li>Projector with minimum 1080p resolution</li>
          <li>Computer with WebGL support</li>
          <li>Node.js 16+ and npm/bun</li>
        </ul>
        <h3>Installation</h3>
        <pre className='bg-gray-100 p-4 rounded-md'>
          <code>
            {`# Clone the repository
git clone https://github.com/your-username/kivy-hub.git

# Install dependencies
cd kivy-hub
bun install

# Start the development server
bun dev`}
          </code>
        </pre>
      </div>
    )
  },
  {
    id: 'mathematics',
    title: 'Mathematics',
    icon: Calculator,
    content: (
      <div className='prose prose-indigo max-w-none'>
        <h2>Mathematical Foundations</h2>
        <p>
          Kivy uses several mathematical concepts to achieve accurate surface
          detection and interaction. Here's a breakdown of the key mathematical
          principles:
        </p>
        <h3>Surface Projection</h3>
        <p>
          The projection mapping is based on homography transformation, which
          maps points from the camera space to the projector space. The
          transformation matrix H is calculated as:
        </p>
        <pre className='bg-gray-100 p-4 rounded-md'>
          <code>
            {`H = [h₁₁ h₁₂ h₁₃]
    [h₂₁ h₂₂ h₂₃]
    [h₃₁ h₃₂ h₃₃]`}
          </code>
        </pre>
        <p>
          Where each point (x, y) in the camera space is transformed to (x', y')
          in the projector space using:
        </p>
        <pre className='bg-gray-100 p-4 rounded-md'>
          <code>
            {`x' = (h₁₁x + h₁₂y + h₁₃) / (h₃₁x + h₃₂y + h₃₃)
y' = (h₂₁x + h₂₂y + h₂₃) / (h₃₁x + h₃₂y + h₃₃)`}
          </code>
        </pre>
        <h3>Hand Tracking</h3>
        <p>
          Hand tracking uses a combination of computer vision techniques and
          geometric transformations to accurately detect and track hand
          movements in 3D space.
        </p>
      </div>
    )
  },
  {
    id: 'ai-model',
    title: 'AI Model',
    icon: Brain,
    content: (
      <div className='prose prose-indigo max-w-none'>
        <h2>AI Model Architecture</h2>
        <p>
          Kivy uses a custom-trained deep learning model for surface detection
          and hand tracking. The model architecture is based on a modified
          version of YOLOv8 with additional layers for surface analysis.
        </p>
        <h3>Model Structure</h3>
        <pre className='bg-gray-100 p-4 rounded-md'>
          <code>
            {`Model Architecture:
Input Layer (640x640x3)
↓
Backbone (CSPDarknet)
↓
Neck (PANet)
↓
Head (Detection + Surface Analysis)
↓
Output Layer`}
          </code>
        </pre>
        <h3>Training</h3>
        <p>
          The model was trained on a custom dataset of over 10,000 images with
          various surfaces and lighting conditions. Training parameters:
        </p>
        <ul>
          <li>Batch size: 16</li>
          <li>Epochs: 100</li>
          <li>Learning rate: 0.001</li>
          <li>Optimizer: Adam</li>
        </ul>
        <h3>Performance</h3>
        <p>The model achieves:</p>
        <ul>
          <li>Surface detection accuracy: 98.5%</li>
          <li>Hand tracking precision: 95.2%</li>
          <li>Average inference time: 16ms</li>
        </ul>
      </div>
    )
  },
  {
    id: 'api',
    title: 'API Reference',
    icon: Code,
    content: (
      <div className='prose prose-indigo max-w-none'>
        <h2>API Reference</h2>
        <p>
          Kivy provides a comprehensive API for developers to integrate and
          extend its functionality.
        </p>
        <h3>Core Components</h3>
        <pre className='bg-gray-100 p-4 rounded-md'>
          <code>
            {`// Initialize Kivy
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

// Handle surface detection
kivy.onSurfaceDetected((surface) => {
  console.log('Surface detected:', surface);
});

// Handle hand tracking
kivy.onHandTracked((hand) => {
  console.log('Hand position:', hand.position);
});`}
          </code>
        </pre>
        <h3>Configuration Options</h3>
        <p>The Kivy instance can be configured with various options:</p>
        <ul>
          <li>Camera settings (resolution, FPS)</li>
          <li>Projector settings (resolution, brightness)</li>
          <li>Tracking sensitivity</li>
          <li>Surface detection parameters</li>
        </ul>
      </div>
    )
  }
];

export function Docs() {
  const [activeSection, setActiveSection] = useState(sections[0].id);

  return (
    <div className='flex h-[calc(100vh-4rem)]'>
      <div className='w-64 bg-white border-r'>
        <div className='p-4'>
          <div className='flex items-center space-x-2 text-indigo-600'>
            <BookOpen className='h-6 w-6' />
            <h1 className='text-xl font-bold'>Documentation</h1>
          </div>
        </div>
        <nav className='mt-4'>
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 text-sm font-medium ${
                activeSection === section.id
                  ? 'bg-indigo-50 text-indigo-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <section.icon className='h-5 w-5' />
              <span>{section.title}</span>
              <ChevronRight className='h-4 w-4 ml-auto' />
            </button>
          ))}
        </nav>
      </div>

      <div className='flex-1 overflow-auto'>
        <div className='max-w-4xl mx-auto p-8'>
          {sections.find((section) => section.id === activeSection)?.content}
        </div>
      </div>
    </div>
  );
}
