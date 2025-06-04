import { Users, Target, Lightbulb, Award } from 'lucide-react';

const stats = [
  { id: 1, name: 'Active Users', value: '1000+' },
  { id: 2, name: 'Interactive Surfaces', value: '5000+' },
  { id: 3, name: 'Countries', value: '20+' },
  { id: 4, name: 'Success Rate', value: '99%' }
];

const values = [
  {
    name: 'Innovation',
    description:
      "Pushing the boundaries of what's possible with interactive technology.",
    icon: Lightbulb
  },
  {
    name: 'Accessibility',
    description:
      'Making advanced technology accessible to everyone, everywhere.',
    icon: Target
  },
  {
    name: 'Community',
    description: 'Building a global community of creators and innovators.',
    icon: Users
  },
  {
    name: 'Excellence',
    description: 'Committed to delivering the highest quality experience.',
    icon: Award
  }
];

export function About() {
  return (
    <div className='bg-white py-24 sm:py-32'>
      <div className='mx-auto max-w-7xl px-6 lg:px-8'>
        <div className='mx-auto max-w-2xl lg:text-center'>
          <h2 className='text-base font-semibold leading-7 text-indigo-600'>
            Our Story
          </h2>
          <p className='mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl'>
            Revolutionizing Surface Interaction
          </p>
          <p className='mt-6 text-lg leading-8 text-gray-600'>
            Kivy was born from a vision to transform how we interact with our
            environment. We believe that any surface can become an interactive
            canvas, opening up new possibilities for creativity, productivity,
            and entertainment.
          </p>
        </div>

        <div className='mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none'>
          <dl className='grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-4'>
            {stats.map((stat) => (
              <div key={stat.id} className='flex flex-col'>
                <dt className='text-base leading-7 text-gray-600'>
                  {stat.name}
                </dt>
                <dd className='order-first text-3xl font-semibold tracking-tight text-gray-900 sm:text-5xl'>
                  {stat.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <div className='mx-auto mt-32 max-w-2xl sm:mt-40 lg:mt-56 lg:max-w-none'>
          <dl className='grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-4'>
            {values.map((value) => (
              <div key={value.name} className='flex flex-col'>
                <dt className='flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900'>
                  <value.icon
                    className='h-5 w-5 flex-none text-indigo-600'
                    aria-hidden='true'
                  />
                  {value.name}
                </dt>
                <dd className='mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600'>
                  <p className='flex-auto'>{value.description}</p>
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <div className='mx-auto mt-32 max-w-2xl lg:text-center'>
          <h2 className='text-base font-semibold leading-7 text-indigo-600'>
            Our Team
          </h2>
          <p className='mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl'>
            The Minds Behind Kivy
          </p>
          <p className='mt-6 text-lg leading-8 text-gray-600'>
            We're a team of passionate engineers, designers, and innovators
            dedicated to pushing the boundaries of interactive technology. Our
            diverse backgrounds and expertise come together to create something
            truly revolutionary.
          </p>
        </div>
      </div>
    </div>
  );
}
