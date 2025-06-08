export default function QuickStartPage() {
  return (
    <div className="prose prose-indigo max-w-none">
      <h1>Quick Start Guide</h1>
      <p>
        Get up and running with Kivy in minutes. This guide will walk you through
        creating your first interactive surface.
      </p>
      <h2>Basic Setup</h2>
      <pre className="rounded-md bg-gray-100 p-4">
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
      <h2>Creating Your First Interaction</h2>
      <p>
        Here's a simple example of creating an interactive button on your surface:
      </p>
      <pre className="rounded-md bg-gray-100 p-4">
        <code>
          {`// Create an interactive button
const button = new InteractiveButton({
  position: { x: 100, y: 100 },
  size: { width: 200, height: 50 },
  label: 'Click Me'
});

// Add click handler
button.onClick(() => {
  console.log('Button clicked!');
});

// Add to surface
kivy.addInteractiveElement(button);`}
        </code>
      </pre>
      <h2>Next Steps</h2>
      <ul>
        <li>Learn about surface calibration</li>
        <li>Explore advanced hand tracking features</li>
        <li>Customize your interactive elements</li>
        <li>Check out the API reference for more options</li>
      </ul>
    </div>
  );
} 