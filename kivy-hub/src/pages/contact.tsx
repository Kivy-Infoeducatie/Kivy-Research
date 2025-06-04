import { Mail, Phone, MapPin, Send } from 'lucide-react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';

const contactSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters'),
  email: z.string().email('Invalid email address'),
  subject: z.string().min(5, 'Subject must be at least 5 characters'),
  message: z.string().min(10, 'Message must be at least 10 characters')
});

type ContactFormData = z.infer<typeof contactSchema>;

const contactInfo = [
  {
    name: 'Email',
    value: 'contact@kivy.dev',
    icon: Mail
  },
  {
    name: 'Phone',
    value: '+1 (555) 123-4567',
    icon: Phone
  },
  {
    name: 'Address',
    value: '123 Innovation Street, Tech City, TC 12345',
    icon: MapPin
  }
];

export function Contact() {
  const {
    register,
    handleSubmit,
    formState: { errors }
  } = useForm<ContactFormData>({
    resolver: zodResolver(contactSchema)
  });

  const onSubmit = (data: ContactFormData) => {
    console.log(data);
  };

  return (
    <div className='bg-white py-24 sm:py-32'>
      <div className='mx-auto max-w-7xl px-6 lg:px-8'>
        <div className='mx-auto max-w-2xl lg:text-center'>
          <h2 className='text-base font-semibold leading-7 text-indigo-600'>
            Contact Us
          </h2>
          <p className='mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl'>
            Get in Touch
          </p>
          <p className='mt-6 text-lg leading-8 text-gray-600'>
            Have questions about Kivy? We're here to help. Reach out to us
            through any of the following channels or fill out the form below.
          </p>
        </div>

        <div className='mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none'>
          <dl className='grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-3'>
            {contactInfo.map((info) => (
              <div key={info.name} className='flex flex-col'>
                <dt className='flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900'>
                  <info.icon
                    className='h-5 w-5 flex-none text-indigo-600'
                    aria-hidden='true'
                  />
                  {info.name}
                </dt>
                <dd className='mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600'>
                  <p className='flex-auto'>{info.value}</p>
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <div className='mx-auto mt-16 max-w-2xl'>
          <form onSubmit={handleSubmit(onSubmit)} className='space-y-6'>
            <div>
              <label
                htmlFor='name'
                className='block text-sm font-medium leading-6 text-gray-900'
              >
                Name
              </label>
              <div className='mt-2'>
                <input
                  type='text'
                  id='name'
                  {...register('name')}
                  className='block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6'
                />
                {errors.name && (
                  <p className='mt-2 text-sm text-red-600'>
                    {errors.name.message}
                  </p>
                )}
              </div>
            </div>

            <div>
              <label
                htmlFor='email'
                className='block text-sm font-medium leading-6 text-gray-900'
              >
                Email
              </label>
              <div className='mt-2'>
                <input
                  type='email'
                  id='email'
                  {...register('email')}
                  className='block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6'
                />
                {errors.email && (
                  <p className='mt-2 text-sm text-red-600'>
                    {errors.email.message}
                  </p>
                )}
              </div>
            </div>

            <div>
              <label
                htmlFor='subject'
                className='block text-sm font-medium leading-6 text-gray-900'
              >
                Subject
              </label>
              <div className='mt-2'>
                <input
                  type='text'
                  id='subject'
                  {...register('subject')}
                  className='block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6'
                />
                {errors.subject && (
                  <p className='mt-2 text-sm text-red-600'>
                    {errors.subject.message}
                  </p>
                )}
              </div>
            </div>

            <div>
              <label
                htmlFor='message'
                className='block text-sm font-medium leading-6 text-gray-900'
              >
                Message
              </label>
              <div className='mt-2'>
                <textarea
                  id='message'
                  rows={4}
                  {...register('message')}
                  className='block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6'
                />
                {errors.message && (
                  <p className='mt-2 text-sm text-red-600'>
                    {errors.message.message}
                  </p>
                )}
              </div>
            </div>

            <div>
              <button
                type='submit'
                className='flex w-full justify-center rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600'
              >
                <Send className='h-5 w-5 mr-2' />
                Send Message
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
