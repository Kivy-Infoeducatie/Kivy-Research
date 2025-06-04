import { ArrowRight, Camera, Projector, Cpu, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';

const features = [
  {
    name: 'Advanced Camera System',
    description: 'High-precision camera tracking for accurate surface detection and interaction.',
    icon: Camera,
  },
  {
    name: 'Smart Projection',
    description: 'Transform any surface into an interactive display with our advanced projection technology.',
    icon: Projector,
  },
  {
    name: 'AI-Powered',
    description: 'Leveraging cutting-edge AI for intelligent surface recognition and interaction.',
    icon: Cpu,
  },
  {
    name: 'Real-time Processing',
    description: 'Instant response and interaction with minimal latency for a seamless experience.',
    icon: Zap,
  },
];

export function Home() {
  return (
    <div className="bg-white">
      {/* Hero section */}
      <div className="relative isolate overflow-hidden">
        <div className="mx-auto max-w-7xl px-6 pb-24 pt-10 sm:pb-32 lg:flex lg:px-8 lg:py-40">
          <div className="mx-auto max-w-2xl flex-shrink-0 lg:mx-0 lg:max-w-xl lg:pt-8">
            <h1 className="mt-10 text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
              Transform Any Surface Into an Interactive Screen
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              Kivy combines advanced camera tracking and projection technology to create an immersive interactive experience on any surface. Powered by cutting-edge AI, it brings your ideas to life.
            </p>
            <div className="mt-10 flex items-center gap-x-6">
              <Link
                to="/playground"
                className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                Try it now
              </Link>
              <Link to="/docs" className="text-sm font-semibold leading-6 text-gray-900">
                Learn more <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Feature section */}
      <div className="mx-auto mt-32 max-w-7xl px-6 sm:mt-56 lg:px-8">
        <div className="mx-auto max-w-2xl lg:text-center">
          <h2 className="text-base font-semibold leading-7 text-indigo-600">Advanced Technology</h2>
          <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Everything you need to create interactive experiences
          </p>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Kivy combines multiple cutting-edge technologies to deliver a seamless and powerful interactive experience.
          </p>
        </div>
        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-4">
            {features.map((feature) => (
              <div key={feature.name} className="flex flex-col">
                <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900">
                  <feature.icon className="h-5 w-5 flex-none text-indigo-600" aria-hidden="true" />
                  {feature.name}
                </dt>
                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600">
                  <p className="flex-auto">{feature.description}</p>
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      {/* CTA section */}
      <div className="relative isolate mt-32 px-6 py-32 sm:mt-56 sm:py-40 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Ready to transform your space?
          </h2>
          <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-gray-600">
            Join the future of interactive technology. Start creating amazing experiences with Kivy today.
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link
              to="/register"
              className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
            >
              Get started
            </Link>
            <Link to="/contact" className="text-sm font-semibold leading-6 text-gray-900">
              Contact us <span aria-hidden="true">→</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
} 