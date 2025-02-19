import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const response = await fetch('http://localhost:5000/api/trading-status');
    const data = await response.json();
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching trading status:', error);
    return NextResponse.json(
      { error: 'Failed to fetch trading status' },
      { status: 500 }
    );
  }
} 