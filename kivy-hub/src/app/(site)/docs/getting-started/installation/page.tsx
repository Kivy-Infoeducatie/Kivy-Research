export default function InstallationPage() {
  return (
    <div className="prose prose-indigo max-w-none">
      <h1>Installation Guide</h1>
      <p>
        Follow these steps to install Kivy and set up your development environment.
      </p>
      <h2>Prerequisites</h2>
      <ul>
        <li>Camera with at least 720p resolution</li>
        <li>Projector with minimum 1080p resolution</li>
        <li>Computer with WebGL support</li>
        <li>Node.js 16+ and npm/bun</li>
      </ul>
      <h2>Installation Steps</h2>
      <pre className="rounded-md bg-gray-100 p-4">
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
      <h2>Configuration</h2>
      <p>
        After installation, you'll need to configure your camera and projector settings.
        Create a <code>.env</code> file in the root directory with the following variables:
      </p>
      <pre className="rounded-md bg-gray-100 p-4">
        <code>
          {`CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
PROJECTOR_WIDTH=1920
PROJECTOR_HEIGHT=1080`}
        </code>
      </pre>
    </div>
  );
} 