import React from 'react';

interface NutritionalInfo {
  calories?: number;
  protein?: number;
  carbs?: number;
  fat?: number;
}

interface RecipeWidgetProps {
  imageUrl?: string;
  title?: string;
  author?: string;
  steps?: string[];
  cookTime?: string;
  nutritionalInfo?: NutritionalInfo;
  weight?: string;
}

export default function RecipeWidget({
  imageUrl = 'https://images.unsplash.com/photo-1588137378633-dea1336ce1e2',
  title = 'Avocado Toast with Lemon',
  author = 'Chef Maria',
  steps = [
    'Toast bread slices until golden brown',
    'Mash ripe avocado and spread on toast',
    'Top with sliced tomatoes and red onions',
    'Squeeze fresh lemon juice over top',
    'Season with salt, pepper, and red pepper flakes'
  ],
  cookTime = '10 min',
  nutritionalInfo = {
    calories: 320,
    protein: 7,
    carbs: 24,
    fat: 21
  },
  weight = '250g'
}: RecipeWidgetProps) {
  return (
    <div className='fixed top-6 left-6'>
      <div className='max-h-[80vh] w-[400px] overflow-hidden overflow-y-auto rounded-lg bg-white shadow-md'>
        <div className='relative h-64 w-full'>
          <img
            src={imageUrl}
            alt={title}
            className='h-full w-full object-cover'
          />
          <div className='absolute top-3 right-3 rounded-full bg-amber-500 px-4 py-2 text-base font-medium text-white'>
            {cookTime}
          </div>
          <div className='absolute right-3 bottom-3 rounded-full bg-blue-500 px-4 py-2 text-base font-medium text-white'>
            {weight}
          </div>
        </div>

        <div className='p-6'>
          <h2 className='mb-3 text-3xl font-bold text-gray-800'>{title}</h2>
          <p className='mb-4 text-lg text-gray-600'>by {author}</p>

          <div className='mb-5 rounded-lg bg-gray-50 p-4'>
            <h3 className='mb-2 text-lg font-semibold text-gray-700'>
              Nutritional Info:
            </h3>
            <div className='grid grid-cols-4 gap-2'>
              <div className='rounded bg-white p-2 text-center shadow-sm'>
                <p className='text-xl font-bold text-gray-800'>
                  {nutritionalInfo.calories}
                </p>
                <p className='text-sm text-gray-600'>calories</p>
              </div>
              <div className='rounded bg-white p-2 text-center shadow-sm'>
                <p className='text-xl font-bold text-gray-800'>
                  {nutritionalInfo.protein}g
                </p>
                <p className='text-sm text-gray-600'>protein</p>
              </div>
              <div className='rounded bg-white p-2 text-center shadow-sm'>
                <p className='text-xl font-bold text-gray-800'>
                  {nutritionalInfo.carbs}g
                </p>
                <p className='text-sm text-gray-600'>carbs</p>
              </div>
              <div className='rounded bg-white p-2 text-center shadow-sm'>
                <p className='text-xl font-bold text-gray-800'>
                  {nutritionalInfo.fat}g
                </p>
                <p className='text-sm text-gray-600'>fat</p>
              </div>
            </div>
          </div>

          <div className='space-y-5'>
            <h3 className='mb-4 text-xl font-semibold text-gray-700'>
              Preparation Steps:
            </h3>
            <div className='space-y-4'>
              {steps.map((step, index) => (
                <div
                  key={index}
                  className='rounded-md border border-gray-100 bg-gray-50 p-4 shadow-sm'
                >
                  <div className='flex items-center'>
                    <span className='mr-3 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-green-500 text-base text-white'>
                      {index + 1}
                    </span>
                    <p className='text-lg text-gray-700'>{step}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
