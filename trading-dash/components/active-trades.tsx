"use client"

import { useState } from "react"
import { ArrowDown, ArrowUp, Clock, Filter, RefreshCw, Loader2 } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useAccountOverview, useActiveTrades } from "@/lib/api"

export function ActiveTrades() {
  const [isRefreshing, setIsRefreshing] = useState(false)
  const { data: accountData, isLoading: isAccountLoading } = useAccountOverview()
  const { trades: activeTrades, isLoading: isTradesLoading, error } = useActiveTrades()

  const handleRefresh = async () => {
    setIsRefreshing(true)
    // In a real implementation, you'd call a refresh function here
    // For now, we'll just simulate a delay
    setTimeout(() => setIsRefreshing(false), 1000)
  }

  // Financial data from real account info or fallback to defaults
  const financialData = {
    maxDrawdown: accountData ? accountData.maxDrawdown : 12.35,
    freeMargin: accountData ? accountData.freeMargin : 3250.75,
    equity: accountData ? accountData.equity : 5325.50,
    balance: accountData ? accountData.balance : 5125.25
  }

  // Loading state for financial cards
  const renderFinancialValue = (isLoading: boolean, label: string, value: string | number | JSX.Element, description: string, color?: string) => {
    if (isLoading) {
      return (
        <>
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            <span className="text-xl font-bold text-muted-foreground">Loading...</span>
          </div>
          <p className="text-xs text-muted-foreground">Loading {label.toLowerCase()}...</p>
        </>
      )
    }
    
    return (
      <>
        <div className={`text-2xl font-bold ${color || ''}`}>{value}</div>
        <p className="text-xs text-muted-foreground">{description}</p>
      </>
    )
  }

  return (
    <div className="space-y-4">
      {/* Financial Information Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Balance</CardTitle>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              className="h-4 w-4 text-muted-foreground"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M16 12h-6.5" />
              <path d="M12.5 8.5v7" />
            </svg>
          </CardHeader>
          <CardContent>
            {renderFinancialValue(
              isAccountLoading,
              "Balance",
              `$${financialData.balance.toFixed(2)}`,
              "Account balance"
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Equity</CardTitle>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              className="h-4 w-4 text-muted-foreground"
            >
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </CardHeader>
          <CardContent>
            {renderFinancialValue(
              isAccountLoading,
              "Equity",
              <>
                <span className={`${
                  financialData.equity > financialData.balance 
                    ? "text-green-500" 
                    : financialData.equity < financialData.balance 
                      ? "text-red-500" 
                      : ""
                }`}>
                  ${financialData.equity.toFixed(2)}
                  {financialData.equity !== financialData.balance && (
                    <span className="text-sm ml-2">
                      ({financialData.equity > financialData.balance ? "+" : ""}
                      ${(financialData.equity - financialData.balance).toFixed(2)})
                    </span>
                  )}
                </span>
              </>,
              "Balance + floating P/L"
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Free Margin</CardTitle>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              className="h-4 w-4 text-muted-foreground"
            >
              <rect width="20" height="14" x="2" y="5" rx="2" />
              <path d="M2 10h20" />
            </svg>
          </CardHeader>
          <CardContent>
            {renderFinancialValue(
              isAccountLoading,
              "Free Margin",
              `$${financialData.freeMargin.toFixed(2)}`,
              "Available for new trades"
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Maximum Drawdown</CardTitle>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              className="h-4 w-4 text-muted-foreground"
            >
              <path d="M23 18l-9.5-9.5-5 5L1 6" />
              <path d="M17 18h6v-6" />
            </svg>
          </CardHeader>
          <CardContent>
            {renderFinancialValue(
              isAccountLoading,
              "Maximum Drawdown",
              `-${financialData.maxDrawdown.toFixed(2)}%`,
              "Largest historical decline",
              "text-amber-500"
            )}
          </CardContent>
        </Card>
      </div>
      
      {/* Active Trades Card */}
      <Card>
        <CardHeader className="flex flex-row items-center">
          <CardTitle>Active Trades</CardTitle>
          <div className="ml-auto flex items-center gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="h-8 gap-1">
                  <Filter className="h-3.5 w-3.5" />
                  <span>Filter</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel>Filter by</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem>All Trades</DropdownMenuItem>
                <DropdownMenuItem>Buy Orders</DropdownMenuItem>
                <DropdownMenuItem>Sell Orders</DropdownMenuItem>
                <DropdownMenuItem>Profitable Trades</DropdownMenuItem>
                <DropdownMenuItem>Losing Trades</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <Button variant="outline" size="sm" className="h-8 gap-1" onClick={handleRefresh}>
              <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
              <span>Refresh</span>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isTradesLoading ? (
            <div className="flex h-40 items-center justify-center">
              <div className="flex flex-col items-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <p className="mt-2 text-sm text-muted-foreground">Loading active trades...</p>
              </div>
            </div>
          ) : error ? (
            <div className="flex h-40 items-center justify-center">
              <div className="text-center">
                <p className="text-sm font-medium text-destructive">Failed to load active trades</p>
                <p className="mt-1 text-xs text-muted-foreground">{error.message}</p>
                <Button variant="outline" size="sm" className="mt-4" onClick={handleRefresh}>
                  Try Again
                </Button>
              </div>
            </div>
          ) : activeTrades.length === 0 ? (
            <div className="flex h-40 items-center justify-center">
              <div className="text-center">
                <p className="text-sm font-medium">No active trades</p>
                <p className="mt-1 text-xs text-muted-foreground">There are currently no open positions</p>
              </div>
            </div>
          ) : (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Entry Price</TableHead>
                    <TableHead>Current Price</TableHead>
                    <TableHead>Stop Loss</TableHead>
                    <TableHead>Take Profit</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead className="text-right">P/L</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {activeTrades.map((trade) => (
                    <TableRow key={trade.id} className="group animate-fadeIn">
                      <TableCell className="font-medium">{trade.symbol}</TableCell>
                      <TableCell>
                        <Badge
                          variant={trade.type === "buy" ? "default" : "secondary"}
                          className={`${trade.type === "buy" ? "bg-green-500" : "bg-red-500"} text-white`}
                        >
                          {trade.type.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell>${trade.entryPrice.toFixed(2)}</TableCell>
                      <TableCell className="relative">
                        <div className="flex items-center">
                          ${trade.currentPrice.toFixed(2)}
                          {trade.currentPrice > trade.entryPrice && trade.type === "buy" ? (
                            <ArrowUp className="ml-1 h-4 w-4 text-green-500" />
                          ) : trade.currentPrice < trade.entryPrice && trade.type === "sell" ? (
                            <ArrowDown className="ml-1 h-4 w-4 text-green-500" />
                          ) : (
                            <ArrowDown className="ml-1 h-4 w-4 text-red-500" />
                          )}
                        </div>
                      </TableCell>
                      <TableCell>${trade.stopLoss.toFixed(2)}</TableCell>
                      <TableCell>${trade.takeProfit.toFixed(2)}</TableCell>
                      <TableCell>
                        <div className="flex items-center">
                          <Clock className="mr-1 h-4 w-4 text-muted-foreground" />
                          {trade.duration}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className={`font-medium ${trade.status === "profit" ? "text-green-500" : "text-red-500"}`}>
                          {trade.status === "profit" ? "+" : "-"}${Math.abs(trade.profitLoss).toFixed(2)} (
                          {trade.status === "profit" ? "+" : "-"}
                          {trade.profitLossPercentage.toFixed(2)}%)
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

