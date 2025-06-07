import Link from 'next/link';
import { BookOpen, Home, LogIn, Play, UserPlus } from 'lucide-react';

const navigation = [
  { name: 'Home', href: '/', icon: Home },
  { name: 'Documentation', href: '/docs', icon: BookOpen },
  { name: 'Playground', href: '/playground', icon: Play }
];

const authNavigation = [
  { name: 'Login', href: '/login', icon: LogIn },
  { name: 'Register', href: '/register', icon: UserPlus }
];

export function Header() {
  return (
    <header className='border-b bg-white'>
      <nav className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
        <div className='flex h-16 items-center justify-between'>
          <div className='flex items-center'>
            <Link href='/' className='flex items-center'>
              <span className='text-2xl font-bold text-indigo-600'>Kivy</span>
            </Link>
          </div>

          <div className='hidden md:block'>
            <div className='ml-10 flex items-center space-x-4'>
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className='flex items-center px-3 py-2 text-sm font-medium text-gray-700 hover:text-indigo-600'
                >
                  <item.icon className='mr-2 h-5 w-5' />
                  {item.name}
                </Link>
              ))}
            </div>
          </div>

          <div className='hidden md:block'>
            <div className='ml-10 flex items-center space-x-4'>
              {authNavigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className='flex items-center px-3 py-2 text-sm font-medium text-gray-700 hover:text-indigo-600'
                >
                  <item.icon className='mr-2 h-5 w-5' />
                  {item.name}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
}
