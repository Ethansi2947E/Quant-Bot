import { useEffect, useState, useRef, useCallback } from 'react';

// Define interface types
export interface IAccountOverview {
  balance: number;
  equity: number;
  freeMargin: number;
  maxDrawdown: number;
  riskLevel: string;
  dailyPnL: number;
  openPositions: number;
  totalTrades?: number;
}

export interface IActiveTrade {
  id: string;
  symbol: string;
  type: string;
  entryPrice: number;
  currentPrice: number;
  stopLoss: number;
  takeProfit: number;
  duration: string;
  status: string;
  profitLoss: number;
  profitLossPercentage: number;
}

export interface ITrade {
  id: string;
  ticket: number;
  symbol: string;
  direction: string;
  entry_price: number;
  exit_price: number;
  volume: number;
  profit: number;
  profit_pct: number;
  open_time: string;
  close_time: string;
  stop_loss: number;
  take_profit: number;
  strategy: string;
  duration: string;
  status: string;
  is_active: boolean;
  
  // Additional properties from backend API
  type?: string;
  entryPrice?: number;
  exitPrice?: number;
  currentPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  profitLoss?: number;
  profitLossPercentage?: number;
  size?: number;
  openTime?: string;
  closeTime?: string;
  isActive?: boolean;
  
  // Snake_case alternatives from database
  profit_loss?: number;
  profit_loss_pips?: number;
  current_price?: number;
  position_size?: number;
}

export interface ITradeHistoryData {
  trades: ITrade[];
  total_count: number;
  summary: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    total_profit: number;
    total_loss: number;
    net_pnl: number;
    win_rate: number;
    average_profit: number;
    average_loss: number;
    profit_factor: number;
    largest_win: number;
    largest_loss: number;
    average_duration: string;
  };
}

export interface IPerformanceMetrics {
  totalTrades: number;
  winRate: number;
  profitLoss: number;
  profitLossPercentage: number;
  portfolioValue: number;
  portfolioGrowth: number;
  timeframe: string;
  chartData: {
    daily: Array<{
      timestamp: string;
      value: number;
      pnl: number;
      winRate: number;
      drawdown: number;
    }>;
    weekly: Array<{
      timestamp: string;
      value: number;
      pnl: number;
      winRate: number;
      drawdown: number;
    }>;
    monthly: Array<{
      timestamp: string;
      value: number;
      pnl: number;
      winRate: number;
      drawdown: number;
    }>;
  };
}

// Add Recent Activity interface
export interface IRecentActivity {
  todayTrades: number;
  weeklyData: Array<{
    day: string;
    trades: number;
    volume: number;
  }>;
}

// WebSocket message types
interface IWebSocketMessage {
  type: string;
  data: any;
}

// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

// WebSocket connection singleton
let socket: WebSocket | null = null;
let socketListeners: { [key: string]: Array<(data: any) => void> } = {};

// Initialize WebSocket connection
export const initializeWebSocket = (): WebSocket => {
  if (socket) return socket;

  socket = new WebSocket(`${WS_BASE_URL}/ws`);

  socket.onopen = () => {
    console.log('WebSocket connection established');
  };

  socket.onmessage = (event) => {
    try {
      const message: IWebSocketMessage = JSON.parse(event.data);
      
      // Dispatch to any registered listeners for this message type
      if (socketListeners[message.type]) {
        socketListeners[message.type].forEach(listener => listener(message.data));
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  };

  socket.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  socket.onclose = () => {
    console.log('WebSocket connection closed');
    // Attempt to reconnect after a delay
    setTimeout(() => {
      socket = null;
      initializeWebSocket();
    }, 5000);
  };

  return socket;
};

// Register a WebSocket message listener
export const subscribeToWebSocketEvents = (
  messageType: string, 
  callback: (data: any) => void
): () => void => {
  if (!socketListeners[messageType]) {
    socketListeners[messageType] = [];
  }
  
  socketListeners[messageType].push(callback);
  
  // Return unsubscribe function
  return () => {
    socketListeners[messageType] = socketListeners[messageType].filter(
      listener => listener !== callback
    );
  };
};

// Hook to use account overview data
export const useAccountOverview = () => {
  const [data, setData] = useState<IAccountOverview | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    // Initialize WebSocket if not already
    const ws = initializeWebSocket();
    
    // Subscribe to account overview updates
    const unsubscribe = subscribeToWebSocketEvents('account_overview', (accountData) => {
      setData(accountData);
      setIsLoading(false);
    });
    
    // Fetch initial data
    const fetchData = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/dashboard/overview`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const jsonData = await response.json();
        setData(jsonData);
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to fetch overview data:', err);
        setError(err instanceof Error ? err : new Error(String(err)));
        setIsLoading(false);
      }
    };

    fetchData();
    
    // Set up polling fallback
    const pollInterval = setInterval(fetchData, 30000); // 30-second refresh
    
    return () => {
      unsubscribe();
      clearInterval(pollInterval);
    };
  }, []);

  return { data, isLoading, error };
};

// Hook to use active trades data
export const useActiveTrades = () => {
  const [trades, setTrades] = useState<IActiveTrade[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    // Initialize WebSocket if not already
    const ws = initializeWebSocket();
    
    // Subscribe to active trades updates
    const unsubscribe = subscribeToWebSocketEvents('active_trades', (tradesData) => {
      setTrades(tradesData);
      setIsLoading(false);
    });
    
    // Fetch initial data
    const fetchData = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/dashboard/active-trades`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const jsonData = await response.json();
        setTrades(jsonData);
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to fetch active trades data:', err);
        setError(err instanceof Error ? err : new Error(String(err)));
        setIsLoading(false);
      }
    };

    fetchData();
    
    // Set up polling fallback
    const pollInterval = setInterval(fetchData, 15000); // 15-second refresh
    
    return () => {
      unsubscribe();
      clearInterval(pollInterval);
    };
  }, []);

  return { trades, isLoading, error };
};

// Hook to use performance metrics data
export const usePerformanceMetrics = (initialTimeframe: string = '1D') => {
  const [metrics, setMetrics] = useState<IPerformanceMetrics | null>(null);
  const [timeframe, setTimeframe] = useState(initialTimeframe); 
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const timeframeRef = useRef(timeframe);

  // Update the ref when timeframe changes
  useEffect(() => {
    timeframeRef.current = timeframe;
  }, [timeframe]);

  // Function to fetch data that can be called from outside
  const fetchData = useCallback(async () => {
    setIsLoading(true);
    try {
      console.log(`Fetching performance metrics with timeframe: ${timeframeRef.current}`);
      
      // First try to fetch from performance endpoint
      const response = await fetch(`${API_BASE_URL}/api/dashboard/performance?timeframe=${timeframeRef.current}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      let jsonData = await response.json();
      
      // If no data is returned or totalTrades is 0, try to fetch from account history
      if (!jsonData || !jsonData.totalTrades || jsonData.totalTrades === 0) {
        console.log("No trade data from performance endpoint, trying account history...");
        try {
          const historyResponse = await fetch(`${API_BASE_URL}/api/dashboard/account-history`);
          
          if (historyResponse.ok) {
            const historyData = await historyResponse.json();
            console.log("Account history data:", historyData);
            
            // Create default structure if jsonData is empty
            if (!jsonData) {
              jsonData = {
                totalTrades: 0,
                winRate: 0,
                profitLoss: 0,
                profitLossPercentage: 0,
                portfolioValue: 0,
                portfolioGrowth: 0,
                timeframe: timeframeRef.current,
                chartData: {
                  daily: [],
                  weekly: [],
                  monthly: []
                }
              };
            }
            
            // Enrich the performance data with history data
            if (historyData && Array.isArray(historyData.trades) && historyData.trades.length > 0) {
              console.log(`Found ${historyData.trades.length} historical trades`);
              
              // Calculate metrics from historical trades
              const totalTrades = historyData.trades.length;
              const winningTrades = historyData.trades.filter((trade: { profit: number, profit_loss?: number }) => 
                (trade.profit > 0 || (trade.profit_loss && trade.profit_loss > 0))
              ).length;
              const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
              
              // Handle different field names
              const profitLoss = historyData.trades.reduce((sum: number, trade: { profit: number, profit_loss?: number }) => 
                sum + (trade.profit_loss !== undefined ? trade.profit_loss : trade.profit), 0
              );
              
              // Merge with existing data
              jsonData.totalTrades = totalTrades;
              jsonData.winRate = winRate;
              jsonData.profitLoss = Number(profitLoss.toFixed(2));
              jsonData.profitLossPercentage = Number((profitLoss / 1000 * 100).toFixed(2)); // Assuming starting balance of 1000
              
              // Use balance history to populate chart data if available
              if (historyData.balance_history && Array.isArray(historyData.balance_history)) {
                // Process balance history for chart data
                console.log(`Found ${historyData.balance_history.length} balance history records`);
                
                // Calculate total profit/loss from balance history
                let totalPL = 0;
                historyData.balance_history.forEach((item: any) => {
                  if (item.profit_loss) {
                    totalPL += Number(item.profit_loss);
                  }
                });
                
                // Use calculated profit/loss if we found values
                if (Math.abs(totalPL) > 0.01) {
                  jsonData.profitLoss = Number(totalPL.toFixed(2));
                  
                  // Calculate percentage based on starting balance
                  const startBalance = historyData.balance_history[0]?.balance || 1000;
                  jsonData.profitLossPercentage = Number((totalPL / startBalance * 100).toFixed(2));
                }
                
                // Last item's balance is current portfolio value if available
                if (historyData.balance_history.length > 0) {
                  const latestBalance = historyData.balance_history[historyData.balance_history.length - 1];
                  jsonData.portfolioValue = latestBalance.balance;
                }
                
                // Format balance history for charts
                const chartEntries = historyData.balance_history.map((item: any) => {
                  const date = new Date(item.date);
                  return {
                    timestamp: date.toISOString(),
                    value: item.balance,
                    pnl: item.profit_loss || item.profit || 0, // Handle different field names
                    winRate: item.win_rate || 0,
                    drawdown: item.drawdown || 0
                  };
                });
                
                // Sort by date
                const sortedEntries = chartEntries.sort(
                  (a: any, b: any) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
                );
                
                // Assign to appropriate timeframes
                jsonData.chartData.daily = sortedEntries;
                
                // For weekly, group by week
                const weeklyData: { [key: string]: any } = {};
                sortedEntries.forEach((entry: any) => {
                  const date = new Date(entry.timestamp);
                  const weekNumber = Math.floor(date.getDate() / 7);
                  const weekKey = `${date.getFullYear()}-${date.getMonth()}-${weekNumber}`;
                  
                  if (!weeklyData[weekKey]) {
                    weeklyData[weekKey] = { ...entry };
                  } else {
                    // Take most recent value for the week
                    if (new Date(entry.timestamp) > new Date(weeklyData[weekKey].timestamp)) {
                      weeklyData[weekKey] = { ...entry };
                    }
                  }
                });
                
                jsonData.chartData.weekly = Object.values(weeklyData);
                
                // For monthly, group by month
                const monthlyData: { [key: string]: any } = {};
                sortedEntries.forEach((entry: any) => {
                  const date = new Date(entry.timestamp);
                  const monthKey = `${date.getFullYear()}-${date.getMonth()}`;
                  
                  if (!monthlyData[monthKey]) {
                    monthlyData[monthKey] = { ...entry };
                  } else {
                    // Take most recent value for the month
                    if (new Date(entry.timestamp) > new Date(monthlyData[monthKey].timestamp)) {
                      monthlyData[monthKey] = { ...entry };
                    }
                  }
                });
                
                jsonData.chartData.monthly = Object.values(monthlyData);
              }
            }
          }
        } catch (historyErr) {
          console.error("Error fetching account history:", historyErr);
        }
      }
      
      console.log("Final metrics data:", jsonData);
      setMetrics(jsonData);
      setIsLoading(false);
    } catch (err) {
      console.error('Failed to fetch performance metrics:', err);
      setError(err instanceof Error ? err : new Error(String(err)));
      setIsLoading(false);
    }
  }, []);

  // Initialize WebSocket for real-time updates
  useEffect(() => {
    const ws = initializeWebSocket();
    
    // Register a listener for performance metrics updates
    const unsubscribe = subscribeToWebSocketEvents('performance_metrics', (data) => {
      setMetrics(data);
      setIsLoading(false);
    });
    
    // Fetch initial data
    fetchData();
    
    // Set up polling fallback with current timeframe
    const pollInterval = setInterval(() => {
      fetchData();
    }, 60000); // 60-second refresh
    
    return () => {
      unsubscribe();
      clearInterval(pollInterval);
    };
  }, [fetchData]);

  // Re-fetch when timeframe changes
  useEffect(() => {
    fetchData();
  }, [timeframe, fetchData]);

  return { 
    metrics, 
    isLoading, 
    error, 
    setTimeframe, 
    refetch: fetchData 
  };
};

// Add useRecentActivity hook
export const useRecentActivity = () => {
  const [activity, setActivity] = useState<IRecentActivity | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    // Initialize WebSocket for real-time updates
    const ws = initializeWebSocket();
    
    // Register a listener for recent activity updates
    const unsubscribe = subscribeToWebSocketEvents('recent_activity', (data) => {
      setActivity(data);
      setIsLoading(false);
    });

    // Initial fetch of recent activity data
    const fetchData = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/dashboard/recent-activity`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch recent activity data: ${response.status}`);
        }
        
        const data = await response.json();
        setActivity(data);
        setIsLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err : new Error(String(err)));
        setIsLoading(false);
      }
    };
    
    fetchData();
    
    return () => {
      unsubscribe();
    };
  }, []);

  return { activity, isLoading, error };
};

// Hook to use trade history data
export const useTradeHistory = (limit: number = 100, offset: number = 0, includeActive: boolean = true) => {
  const [tradeHistory, setTradeHistory] = useState<ITradeHistoryData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  // Function to process trade history data
  const processTradeHistoryData = (data: any): ITradeHistoryData => {
    console.log("Raw API response:", JSON.stringify(data, null, 2));
    
    // Create a default summary if missing or if there are data issues
    if (!data.summary || Object.keys(data.summary).length === 0) {
      data.summary = {
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        total_profit: 0,
        total_loss: 0,
        net_pnl: 0,
        win_rate: 0,
        average_profit: 0,
        average_loss: 0,
        profit_factor: 0,
        largest_win: 0,
        largest_loss: 0,
        average_duration: "0h 0m"
      };
    }
    
    // Process trades data
    if (data && data.trades && Array.isArray(data.trades)) {
      // Count trades for stats calculation
      let totalTrades = data.trades.length;
      let winningTrades = 0;
      let losingTrades = 0;
      let totalProfit = 0;
      let totalLoss = 0;
      
      data.trades = data.trades.map((trade: any) => {
        // Normalize profit value
        const profitValue = processNumericValue(trade.profit || trade.profitLoss || trade.profit_loss);
        
        // Track wins/losses for summary recalculation
        if (profitValue > 0) {
          winningTrades++;
          totalProfit += profitValue;
        } else if (profitValue < 0) {
          losingTrades++;
          totalLoss += Math.abs(profitValue);
        }
        
        // Ensure all required fields have values
        const processedTrade = {
          ...trade,
          id: trade.id?.toString() || trade.ticket?.toString() || Math.random().toString(36).substring(2),
          ticket: parseInt(String(trade.ticket || 0)),
          direction: processTradeDirection(trade),
          symbol: trade.symbol || "Unknown",
          profit: profitValue,
          profit_pct: processNumericValue(trade.profit_pct || trade.profitLossPercentage || trade.profit_loss_pips),
          entry_price: processNumericValue(trade.entry_price || trade.entryPrice),
          exit_price: processNumericValue(trade.exit_price || trade.exitPrice || trade.current_price),
          volume: processNumericValue(trade.volume || trade.size || trade.position_size),
          stop_loss: processNumericValue(trade.stop_loss || trade.stopLoss),
          take_profit: processNumericValue(trade.take_profit || trade.takeProfit),
          strategy: trade.strategy || "Unknown",
          open_time: trade.open_time || trade.openTime || "",
          close_time: trade.close_time || trade.closeTime || "Active",
          is_active: !!trade.is_active || !!trade.isActive || trade.status === "open"
        };
        
        // Determine status based on profit if not provided
        if (!trade.status) {
          if (processedTrade.profit > 0) {
            processedTrade.status = "profit";
          } else if (processedTrade.profit < 0) {
            processedTrade.status = "loss";
          } else {
            processedTrade.status = "breakeven";
          }
        } else {
          // Map 'closed' status to profit/loss if profit value exists
          if (trade.status === 'closed' && profitValue !== 0) {
            processedTrade.status = profitValue > 0 ? "profit" : "loss";
          } else {
            processedTrade.status = trade.status;
          }
        }
        
        return processedTrade;
      });
      
      // Recalculate summary data
      const netPnl = totalProfit - totalLoss;
      const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
      const avgProfit = winningTrades > 0 ? totalProfit / winningTrades : 0;
      const avgLoss = losingTrades > 0 ? totalLoss / losingTrades : 0;
      const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : 0;
      
      // Update summary with recalculated values
      data.summary = {
        ...data.summary,
        total_trades: processIntValue(data.summary.total_trades) || totalTrades,
        winning_trades: processIntValue(data.summary.winning_trades) || winningTrades,
        losing_trades: processIntValue(data.summary.losing_trades) || losingTrades,
        total_profit: processNumericValue(data.summary.total_profit) || totalProfit,
        total_loss: processNumericValue(data.summary.total_loss) || totalLoss,
        net_pnl: processNumericValue(data.summary.net_pnl) || netPnl,
        win_rate: processNumericValue(data.summary.win_rate) || winRate,
        average_profit: processNumericValue(data.summary.average_profit) || avgProfit,
        average_loss: processNumericValue(data.summary.average_loss) || avgLoss,
        profit_factor: processNumericValue(data.summary.profit_factor) || profitFactor,
        largest_win: processNumericValue(data.summary.largest_win),
        largest_loss: processNumericValue(data.summary.largest_loss),
        average_duration: data.summary.average_duration || "0h 0m"
      };
      
      console.log("Processed trades:", JSON.stringify(data.trades.slice(0, 2), null, 2));
    }
    
    console.log("Processed summary:", JSON.stringify(data.summary, null, 2));
    return data as ITradeHistoryData;
  };
  
  // Helper function to process numeric values
  const processNumericValue = (value: any): number => {
    if (value === undefined || value === null) return 0;
    
    try {
      // Convert to string first to handle various formats
      const stringValue = String(value).trim();
      if (stringValue === '') return 0;
      
      const numValue = parseFloat(stringValue);
      return isNaN(numValue) ? 0 : numValue;
    } catch {
      return 0;
    }
  };
  
  // Helper function to process integer values
  const processIntValue = (value: any): number => {
    if (value === undefined || value === null) return 0;
    
    try {
      // Convert to string first to handle various formats
      const stringValue = String(value).trim();
      if (stringValue === '') return 0;
      
      const numValue = parseInt(stringValue, 10);
      return isNaN(numValue) ? 0 : numValue;
    } catch {
      return 0;
    }
  };
  
  // Helper function to process trade direction
  const processTradeDirection = (trade: any): string => {
    // First try to get direct direction field (this is what's in the database)
    if (typeof trade.direction === 'string' && trade.direction.trim() !== '') {
      return trade.direction.toLowerCase();
    }
    
    // Fall back to type field (used in mock data)
    if (typeof trade.type === 'string' && trade.type.trim() !== '') {
      return trade.type.toLowerCase();
    }
    
    // Use profit as hint if direction is missing
    const profit = processNumericValue(trade.profit_loss || trade.profit || trade.profitLoss);
    if (profit !== 0) {
      // This is a guess - in real trading many other factors affect profit direction
      return profit > 0 ? 'buy' : 'sell';
    }
    
    return "unknown";
  };

  // Function to fetch trade history - can be called again for pagination
  const fetchTradeHistory = useCallback(async (newLimit?: number, newOffset?: number, newIncludeActive?: boolean) => {
    setIsLoading(true);
    
    try {
      // Use current values as defaults if new ones aren't provided
      const queryLimit = newLimit !== undefined ? newLimit : limit;
      const queryOffset = newOffset !== undefined ? newOffset : offset;
      const queryIncludeActive = newIncludeActive !== undefined ? newIncludeActive : includeActive;
      
      // Build query string
      const queryParams = new URLSearchParams({
        limit: queryLimit.toString(),
        offset: queryOffset.toString(),
        include_active: queryIncludeActive.toString()
      });
      
      const response = await fetch(`${API_BASE_URL}/api/dashboard/trade-history?${queryParams}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch trade history: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Process the data
      const processedData = processTradeHistoryData(data);
      
      setTradeHistory(processedData);
      setIsLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setIsLoading(false);
    }
  }, [limit, offset, includeActive]);

  useEffect(() => {
    // Initialize WebSocket for real-time updates
    const ws = initializeWebSocket();
    
    // Register a listener for trade history updates
    const unsubscribe = subscribeToWebSocketEvents('trade_history', (data) => {
      // Process the data
      const processedData = processTradeHistoryData(data);
      
      setTradeHistory(processedData);
      setIsLoading(false);
    });

    // Initial fetch of trade history data
    fetchTradeHistory();
    
    return () => {
      unsubscribe();
    };
  }, [fetchTradeHistory]);

  return { tradeHistory, isLoading, error, fetchTradeHistory };
};