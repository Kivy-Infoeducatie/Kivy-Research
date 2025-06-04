import './App.css';
import './lib/fontawesome/css/fa.css';
import { useHandTracking } from './lib/hand-tracking/hand-tracking-context.tsx';
import { HandTrackingVideo } from './components/dev/hand-tracking-video.tsx';
import { HandCursor } from './components/dev/hand-cursor.tsx';
import Selectable from './components/core/selectable.tsx';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { RootLayout } from './components/layout/root-layout.tsx';
import { Home } from './pages/home.tsx';
import { About } from './pages/about.tsx';
import { Contact } from './pages/contact.tsx';
import { Docs } from './pages/docs.tsx';
import { Login } from './pages/login.tsx';
import { Register } from './pages/register.tsx';

function Playground() {
  const { toggleTracking } = useHandTracking();

  return (
    <div>
      <HandTrackingVideo />
      <button onClick={toggleTracking}>Toggle</button>
      <HandCursor />
      <Selectable>
        <div className='bg-white p-4 rounded shadow-md'>
          <h2 className='text-lg font-bold'>Widget</h2>
          <p>This is a movable widget.</p>
        </div>
      </Selectable>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<RootLayout />}>
          <Route path='/' element={<Home />} />
          <Route path='/about' element={<About />} />
          <Route path='/contact' element={<Contact />} />
          <Route path='/docs' element={<Docs />} />
          <Route path='/login' element={<Login />} />
          <Route path='/register' element={<Register />} />
        </Route>
        <Route path='/playground' element={<Playground />} />
      </Routes>
    </BrowserRouter>
  );
}
