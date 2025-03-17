"use client"

import { useState, useEffect } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChevronDown, ChevronLeft, ChevronRight, Search } from "lucide-react"
import { ITrade, ITradeHistoryData, useTradeHistory } from "@/lib/api"
import { Skeleton } from "@/components/ui/skeleton"

export function TradeHistoryTable() {
  const [searchTerm, setSearchTerm] = useState("")
  const [sortBy, setSortBy] = useState("Date (Newest)")
  const [currentPage, setCurrentPage] = useState(1)
  const [includeActive, setIncludeActive] = useState(true)
  const pageSize = 10
  
  // Use the real data hook
  const { tradeHistory, isLoading, error, fetchTradeHistory } = useTradeHistory(
    pageSize,
    (currentPage - 1) * pageSize,
    includeActive
  )
  
  // Debug: Log the raw trade history data when it changes
  useEffect(() => {
    if (tradeHistory) {
      console.log("Raw trade history:", JSON.stringify(tradeHistory, null, 2));
      
      // Debug summary data specifically
      console.log("Summary data:", JSON.stringify(tradeHistory.summary, null, 2));
      
      // Log the first trade to examine its structure
      if (tradeHistory.trades && tradeHistory.trades.length > 0) {
        console.log("Sample trade data:", JSON.stringify(tradeHistory.trades[0], null, 2));
      }
    }
  }, [tradeHistory]);
  
  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage)
    fetchTradeHistory(pageSize, (newPage - 1) * pageSize, includeActive)
  }

  const formatDate = (dateString: string) => {
    if (!dateString || dateString === "Active") return "Active"
    try {
      // Check if it's a valid date
      const timestamp = Date.parse(dateString)
      if (isNaN(timestamp)) return "Active"
      
    const date = new Date(dateString)
    return date.toLocaleString()
    } catch (e) {
      return "Active"
    }
  }
  
  // Format trades data for display
  const formatTradeData = (trades: ITrade[] | undefined): ITrade[] => {
    if (!trades) return []
    
    const formattedTrades = trades.map(trade => {
      // Debug each trade before processing
      console.log(`Processing trade ${trade.id}:`, JSON.stringify(trade, null, 2));
      
      // Normalize direction field (check both type and direction fields)
      let direction = "unknown";
      if (typeof trade.type === 'string' && trade.type.trim() !== '') {
        direction = trade.type.toLowerCase();
      } else if (typeof trade.direction === 'string' && trade.direction.trim() !== '') {
        direction = trade.direction.toLowerCase();
      }
      
      // Force direction to either 'buy' or 'sell' if entry/exit prices hint at direction
      if (direction === 'unknown') {
        const entryPrice = parseFloat(String(trade.entry_price || trade.entryPrice || 0));
        const exitPrice = parseFloat(String(trade.exit_price || trade.exitPrice || 0));
        
        if (entryPrice > 0 && exitPrice > 0) {
          // If exit > entry, likely a buy; if entry > exit, likely a sell
          direction = exitPrice > entryPrice ? 'buy' : 'sell';
        }
      }
      
      // Normalize profit values
      const profit = typeof trade.profit !== 'undefined' ? parseFloat(String(trade.profit)) : 
                    typeof trade.profitLoss !== 'undefined' ? parseFloat(String(trade.profitLoss)) : 
                    typeof trade.profit_loss !== 'undefined' ? parseFloat(String(trade.profit_loss)) : 0;
      
      // Map database status to UI status
      let displayStatus = trade.status || 'unknown';
      if (displayStatus === 'open') {
        displayStatus = 'open';
      } else if (displayStatus === 'closed') {
        // For closed trades, determine status based on profit
        if (profit > 0) {
          displayStatus = 'profit';
        } else if (profit < 0) {
          displayStatus = 'loss'; 
        } else {
          displayStatus = 'breakeven';
        }
      }
      
      const profitPct = typeof trade.profit_pct !== 'undefined' ? parseFloat(String(trade.profit_pct)) : 
                      typeof trade.profitLossPercentage !== 'undefined' ? parseFloat(String(trade.profitLossPercentage)) : 
                      typeof trade.profit_loss_pips !== 'undefined' ? parseFloat(String(trade.profit_loss_pips)) : 0;
      
      const formattedTrade = {
      ...trade,
        id: trade.id?.toString() || Math.random().toString(36).substring(2),
        direction: direction,
        entry_price: typeof trade.entry_price !== 'undefined' ? parseFloat(String(trade.entry_price)) : 
                    typeof trade.entryPrice !== 'undefined' ? parseFloat(String(trade.entryPrice)) : 0,
        exit_price: typeof trade.exit_price !== 'undefined' ? parseFloat(String(trade.exit_price)) : 
                   typeof trade.exitPrice !== 'undefined' ? parseFloat(String(trade.exitPrice)) : 
                   typeof trade.current_price !== 'undefined' ? parseFloat(String(trade.current_price)) : 0,
        volume: typeof trade.volume !== 'undefined' ? parseFloat(String(trade.volume)) : 
               typeof trade.size !== 'undefined' ? parseFloat(String(trade.size)) : 
               typeof trade.position_size !== 'undefined' ? parseFloat(String(trade.position_size)) : 0,
        profit: profit,
        profit_pct: profitPct,
        open_time: trade.open_time || trade.openTime || "",
        close_time: trade.close_time || trade.closeTime || "Active",
      symbol: trade.symbol || "Unknown",
      strategy: trade.strategy || "Unknown",
        status: displayStatus,
        is_active: trade.is_active || trade.isActive || displayStatus === 'open'
      };
      
      // Debug the processed trade
      console.log(`Formatted trade ${formattedTrade.id}:`, JSON.stringify(formattedTrade, null, 2));
      
      return formattedTrade;
    });
    
    return formattedTrades;
  }
  
  const trades = formatTradeData(tradeHistory?.trades)

  // Filter trades based on search term
  const filteredTrades = trades.filter(
    (trade) =>
      trade.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      trade.direction.toLowerCase().includes(searchTerm.toLowerCase()) ||
      trade.strategy.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // Sort filtered trades
  const sortedTrades = [...filteredTrades].sort((a, b) => {
    switch (sortBy) {
      case "Date (Newest)":
        if (a.open_time === "Active" || !a.open_time) return -1
        if (b.open_time === "Active" || !b.open_time) return 1
        try {
        return new Date(b.open_time).getTime() - new Date(a.open_time).getTime()
        } catch (e) {
          return 0
        }
      case "Date (Oldest)":
        if (a.open_time === "Active" || !a.open_time) return 1
        if (b.open_time === "Active" || !b.open_time) return -1
        try {
        return new Date(a.open_time).getTime() - new Date(b.open_time).getTime()
        } catch (e) {
          return 0
        }
      case "Profit (Highest)":
        return (b.profit || 0) - (a.profit || 0)
      case "Profit (Lowest)":
        return (a.profit || 0) - (b.profit || 0)
      default:
        return 0
    }
  })
  
  // Get total pages for pagination
  const totalCount = tradeHistory?.total_count || 0
  const totalPages = Math.ceil(totalCount / pageSize)

  // Show loading state
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-96 w-full" />
      </div>
    )
  }

  // Show error state
  if (error) {
    return (
      <div className="p-4 text-center text-red-500">
        Error loading trade history: {error.message}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Statistics */}
      {tradeHistory?.summary && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Win Rate:</span>
                  <span className="font-medium">{(tradeHistory.summary.win_rate || 0).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Profit Factor:</span>
                  <span className="font-medium">{(tradeHistory.summary.profit_factor || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Net P&L:</span>
                  <span className={`font-medium ${(tradeHistory.summary.net_pnl || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {(tradeHistory.summary.net_pnl || 0).toFixed(2)}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Trades</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Trades:</span>
                  <span className="font-medium">{tradeHistory.summary.total_trades || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Winning Trades:</span>
                  <span className="font-medium text-green-500">{tradeHistory.summary.winning_trades || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Losing Trades:</span>
                  <span className="font-medium text-red-500">{tradeHistory.summary.losing_trades || 0}</span>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Avg. Win:</span>
                  <span className="font-medium text-green-500">{(tradeHistory.summary.average_profit || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Avg. Loss:</span>
                  <span className="font-medium text-red-500">{(tradeHistory.summary.average_loss || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Avg. Duration:</span>
                  <span className="font-medium">{tradeHistory.summary.average_duration || "0h 0m"}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      
      {/* Search and filters */}
      <div className="flex items-center justify-between">
        <div className="relative w-64">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search trades..."
            className="pl-8"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setIncludeActive(!includeActive)
              fetchTradeHistory(pageSize, (currentPage - 1) * pageSize, !includeActive)
            }}
          >
            {includeActive ? "Hide Active Trades" : "Show Active Trades"}
          </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" className="gap-1">
              <span>Sort by: {sortBy}</span>
              <ChevronDown className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
              <DropdownMenuLabel>Sort by</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => setSortBy("Date (Newest)")}>Date (Newest)</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setSortBy("Date (Oldest)")}>Date (Oldest)</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy("Profit (Highest)")}>
                Profit (Highest)
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy("Profit (Lowest)")}>
                Profit (Lowest)
              </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        </div>
      </div>
      
      {/* Trade history table */}
      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead>Direction</TableHead>
              <TableHead>Open Time</TableHead>
              <TableHead>Close Time</TableHead>
              <TableHead>Entry</TableHead>
              <TableHead>Exit</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>P/L</TableHead>
              <TableHead>P/L %</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Strategy</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedTrades.length === 0 ? (
              <TableRow>
                <TableCell colSpan={11} className="h-24 text-center">
                  No trade history found
                </TableCell>
              </TableRow>
            ) : (
              sortedTrades.map((trade) => (
              <TableRow key={trade.id}>
                <TableCell className="font-medium">{trade.symbol}</TableCell>
                <TableCell>
                  <Badge
                      variant="outline"
                      className={`${
                      trade.direction.toLowerCase() === "buy" ? "text-green-500 border-green-200" : 
                      trade.direction.toLowerCase() === "sell" ? "text-red-500 border-red-200" :
                      "text-gray-500 border-gray-200"
                      }`}
                    >
                    {trade.direction ? trade.direction.toUpperCase() : "UNKNOWN"}
                    </Badge>
                  </TableCell>
                  <TableCell>{formatDate(trade.open_time)}</TableCell>
                  <TableCell>{formatDate(trade.close_time)}</TableCell>
                  <TableCell>{(trade.entry_price || 0).toFixed(5)}</TableCell>
                  <TableCell>{(trade.exit_price || 0).toFixed(5)}</TableCell>
                  <TableCell>{(trade.volume || 0).toFixed(2)}</TableCell>
                  <TableCell
                    className={`${trade.profit > 0 ? "text-green-500" : trade.profit < 0 ? "text-red-500" : ""}`}
                  >
                    {(trade.profit || 0).toFixed(2)}
                  </TableCell>
                  <TableCell
                    className={`${trade.profit_pct > 0 ? "text-green-500" : trade.profit_pct < 0 ? "text-red-500" : ""}`}
                  >
                    {(trade.profit_pct || 0).toFixed(2)}%
                  </TableCell>
                  <TableCell>
                    <Badge
                    variant={
                      trade.status === "profit" ? "default" : 
                      trade.status === "loss" ? "destructive" :
                      trade.status === "open" ? "outline" :
                      "secondary"
                    }
                    className={
                      trade.status === "open" ? "text-blue-500 border-blue-200" : ""
                    }
                  >
                    {trade.status === "profit" ? "Win" : 
                     trade.status === "loss" ? "Loss" : 
                     trade.status === "open" ? "Active" :
                     "Even"}
                  </Badge>
                </TableCell>
                  <TableCell>{trade.strategy}</TableCell>
              </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
      
      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Showing {(currentPage - 1) * pageSize + 1} to {Math.min(currentPage * pageSize, totalCount)} of {totalCount} entries
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <div className="text-sm">
              Page {currentPage} of {totalPages}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

