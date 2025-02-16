import { NextResponse } from 'next/server';

export async function POST() {
  try {
    const response = await fetch('http://localhost:5000/api/toggle-trading', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error('Flask server error:', errorData);
      throw new Error(`Flask server error: ${errorData}`);
    }

    const data = await response.json();
    
    return new NextResponse(JSON.stringify(data), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST',
      },
    });
  } catch (error: any) {
    console.error('Error toggling trading status:', error);
    return new NextResponse(
      JSON.stringify({ error: error?.message || 'Failed to toggle trading status' }),
      { 
        status: 500,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      }
    );
  }
} 