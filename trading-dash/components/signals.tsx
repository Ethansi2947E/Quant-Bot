"use client"

import { useState, useEffect } from "react"
import { Info, RefreshCw, Search } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { subscribeToWebSocketEvents, initializeWebSocket } from "@/lib/api"

// API base URL from environment
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Signal interface
interface ISignal {
  id: string;
  timestamp: string;
  symbol: string;
  type: string;
  price: number;
  confidence: string;
  explanation: string;
}

// Custom hook for signals data
function useSignals() {
  const [signals, setSignals] = useState<ISignal[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchSignals = async () => {
    setIsLoading(true);
    try {
      // Use API_BASE_URL instead of relative URL
      const response = await fetch(`${API_BASE_URL}/api/dashboard/signals`);
      
      if (!response.ok) {
        // If we get a 404 or other error, fall back to mock data
        console.warn("Failed to fetch signals, using mock data");
        setSignals(mockSignals);
        return;
      }
      
      const data = await response.json();
      
      // Check if data is an array and has items
      if (Array.isArray(data) && data.length > 0) {
        setSignals(data);
      } else {
        console.log("No signals available from API, using mock data");
        setSignals(mockSignals);
      }
    } catch (err) {
      console.error("Error fetching signals:", err);
      setError(err instanceof Error ? err : new Error(String(err)));
      // Fall back to mock data on error
      setSignals(mockSignals);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Initialize WebSocket
    const ws = initializeWebSocket();
    
    // Subscribe to signal updates
    const unsubscribe = subscribeToWebSocketEvents('new_signal', (signal) => {
      setSignals(prev => [signal, ...prev]);
    });
    
    // Fetch initial data
    fetchSignals();
    
    return () => {
      unsubscribe();
    };
  }, []);

  return {
    signals,
    isLoading,
    error,
    refetch: fetchSignals
  };
}

// Mock data for signals (fallback)
const mockSignals: ISignal[] = [
  {
    id: "1",
    timestamp: "2023-06-15T14:30:00Z",
    symbol: "BTC/USD",
    type: "buy",
    price: 42350.75,
    confidence: "high",
    explanation:
      "Strong bullish divergence on the 4-hour RSI, with price breaking above the 50-day moving average. Volume increasing with price action, suggesting strong buying pressure. MACD showing bullish crossover.",
  },
  {
    id: "2",
    timestamp: "2023-06-15T12:15:00Z",
    symbol: "ETH/USD",
    type: "buy",
    price: 2250.5,
    confidence: "medium",
    explanation:
      "Price bounced off major support level with increased volume. Stochastic RSI showing oversold conditions with potential reversal. ETH/BTC ratio improving, suggesting relative strength.",
  },
  {
    id: "3",
    timestamp: "2023-06-15T10:45:00Z",
    symbol: "SOL/USD",
    type: "sell",
    price: 105.25,
    confidence: "high",
    explanation:
      "Double top formation confirmed with price breaking below neckline. Decreasing volume on recent rallies suggests weakening buying pressure. RSI showing bearish divergence on multiple timeframes.",
  },
  {
    id: "4",
    timestamp: "2023-06-15T09:20:00Z",
    symbol: "XRP/USD",
    type: "buy",
    price: 0.5125,
    confidence: "low",
    explanation:
      "Price approaching long-term trendline support. Bollinger Bands showing potential squeeze, indicating upcoming volatility. Low confidence due to overall market uncertainty and regulatory concerns.",
  },
  {
    id: "5",
    timestamp: "2023-06-15T08:00:00Z",
    symbol: "ADA/USD",
    type: "sell",
    price: 0.385,
    confidence: "medium",
    explanation:
      "Head and shoulders pattern forming on the daily chart. Volume increasing on downward moves. 20-day EMA crossing below 50-day EMA, suggesting bearish momentum.",
  },
  {
    id: "6",
    timestamp: "2023-06-15T06:30:00Z",
    symbol: "DOGE/USD",
    type: "buy",
    price: 0.0725,
    confidence: "medium",
    explanation:
      "Bullish engulfing pattern on the daily chart. Social sentiment analysis showing increasing positive mentions. Price breaking above descending trendline with increased volume.",
  },
  {
    id: "7",
    timestamp: "2023-06-15T05:15:00Z",
    symbol: "LINK/USD",
    type: "buy",
    price: 15.75,
    confidence: "high",
    explanation:
      "Strong accumulation pattern with increasing on-chain activity. Price consolidating above major support level. Relative strength compared to other altcoins during market weakness.",
  },
]

export function Signals() {
  const { signals, isLoading, refetch } = useSignals();
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedSignal, setSelectedSignal] = useState<ISignal | null>(null)

  const handleRefresh = () => {
    setIsRefreshing(true)
    refetch();
    setTimeout(() => setIsRefreshing(false), 1000)
  }

  const filteredSignals = signals.filter(
    (signal) =>
      signal.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      signal.type.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleString()
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center">
        <CardTitle>Trading Signals</CardTitle>
        <div className="ml-auto flex items-center gap-2">
          <div className="relative w-40 md:w-60">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search signals..."
              className="pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <Button variant="outline" size="sm" className="h-8 gap-1" onClick={handleRefresh}>
            <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex h-[300px] w-full items-center justify-center">
            <RefreshCw className="h-6 w-6 animate-spin text-primary" />
            <span className="ml-2 text-lg">Loading signals...</span>
          </div>
        ) : (
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Price</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead className="text-right">Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredSignals.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} className="h-24 text-center">
                      No signals found. {searchTerm ? "Try a different search term." : "New signals will appear here when generated."}
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredSignals.map((signal) => (
                    <TableRow key={signal.id} className="group animate-fadeIn">
                      <TableCell>{formatDate(signal.timestamp)}</TableCell>
                      <TableCell className="font-medium">{signal.symbol}</TableCell>
                      <TableCell>
                        <Badge
                          variant={signal.type === "buy" ? "default" : "secondary"}
                          className={`${signal.type === "buy" ? "bg-green-500" : "bg-red-500"} text-white`}
                        >
                          {signal.type.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell>${signal.price.toFixed(2)}</TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={`${
                            signal.confidence === "high"
                              ? "border-green-500 text-green-500"
                              : signal.confidence === "medium"
                                ? "border-yellow-500 text-yellow-500"
                                : "border-red-500 text-red-500"
                          }`}
                        >
                          {signal.confidence.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button variant="ghost" size="icon" onClick={() => setSelectedSignal(signal)}>
                              <Info className="h-4 w-4" />
                              <span className="sr-only">View signal details</span>
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="sm:max-w-md">
                            <DialogHeader>
                              <DialogTitle>
                                {signal.type.toUpperCase()} Signal for {signal.symbol}
                              </DialogTitle>
                              <DialogDescription>Generated on {formatDate(signal.timestamp)}</DialogDescription>
                            </DialogHeader>
                            <div className="space-y-4 py-4">
                              <div className="grid grid-cols-2 gap-4">
                                <div>
                                  <h4 className="text-sm font-medium text-muted-foreground">Signal Type</h4>
                                  <p className="font-medium">{signal.type.toUpperCase()}</p>
                                </div>
                                <div>
                                  <h4 className="text-sm font-medium text-muted-foreground">Price</h4>
                                  <p className="font-medium">${signal.price.toFixed(2)}</p>
                                </div>
                                <div>
                                  <h4 className="text-sm font-medium text-muted-foreground">Confidence</h4>
                                  <p className="font-medium">{signal.confidence.toUpperCase()}</p>
                                </div>
                                <div>
                                  <h4 className="text-sm font-medium text-muted-foreground">Symbol</h4>
                                  <p className="font-medium">{signal.symbol}</p>
                                </div>
                              </div>
                              <div>
                                <h4 className="text-sm font-medium text-muted-foreground">Signal Explanation</h4>
                                <p className="mt-1 text-sm">{signal.explanation}</p>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

