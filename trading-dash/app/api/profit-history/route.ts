import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  try {
    // Get the timeframe from the URL query parameters
    const { searchParams } = new URL(request.url);
    const timeframe = searchParams.get('timeframe') || '7D';

    // Try to fetch from the trading bot's API
    try {
      const response = await fetch(`http://localhost:5000/api/profit-history?timeframe=${timeframe}`);
      if (!response.ok) {
        throw new Error('Failed to fetch from trading bot API');
      }
      const data = await response.json();
      
      // Validate the data structure
      if (!data || !Array.isArray(data.data)) {
        throw new Error('Invalid data structure from trading bot API');
      }
      
      return NextResponse.json({
        data: data.data.map((entry: any) => ({
          timestamp: entry.timestamp,
          profit: Number(entry.profit),
          trade_type: entry.trade_type || null,
          cumulative: Number(entry.cumulative)
        })),
        cumulative_profit: Number(data.cumulative_profit),
        timeframe: timeframe
      });
    } catch (error) {
      console.warn('Failed to fetch from trading bot API, using mock data:', error);
      
      // Generate mock data if the trading bot API is not available
      const now = new Date();
      const mockData = {
        data: Array.from({ length: 24 }, (_, i) => {
          const timestamp = new Date(now.getTime() - (23 - i) * 3600000).toISOString();
          const profit = (Math.random() * 200 - 100).toFixed(2);
          const cumulative = (1000 + Math.random() * 500).toFixed(2);
          return {
            timestamp,
            profit: Number(profit),
            trade_type: Math.random() > 0.5 ? 'BUY' : 'SELL',
            cumulative: Number(cumulative)
          };
        }),
        cumulative_profit: 1500.50,
        timeframe: timeframe
      };
      
      return NextResponse.json(mockData);
    }
  } catch (error) {
    console.error('Error in profit history API route:', error);
    return NextResponse.json(
      { error: 'Failed to fetch profit history data' },
      { status: 500 }
    );
  }
} 