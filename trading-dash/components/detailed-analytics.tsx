"use client"

import { useState } from "react"
import { CalendarIcon, ChevronDown, Filter } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { format } from "date-fns"
import { PerformanceChart } from "@/components/performance-chart"
import { WinLossChart } from "@/components/win-loss-chart"
import { TradeHistoryTable } from "@/components/trade-history-table"

export function DetailedAnalytics() {
  const [date, setDate] = useState<Date>()
  const [asset, setAsset] = useState<string>("All Assets")

  return (
    <Card>
      <CardHeader className="flex flex-row items-center">
        <CardTitle>Detailed Analytics</CardTitle>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 gap-1">
                <span>{asset}</span>
                <ChevronDown className="h-3.5 w-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Select Asset</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setAsset("All Assets")}>All Assets</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setAsset("BTC/USD")}>BTC/USD</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setAsset("ETH/USD")}>ETH/USD</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setAsset("SOL/USD")}>SOL/USD</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setAsset("XRP/USD")}>XRP/USD</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 gap-1 w-[240px] justify-start text-left font-normal">
                <CalendarIcon className="h-3.5 w-3.5" />
                {date ? format(date, "PPP") : <span>Pick a date</span>}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0">
              <Calendar mode="single" selected={date} onSelect={setDate} initialFocus />
            </PopoverContent>
          </Popover>
          <Button variant="outline" size="sm" className="h-8 gap-1">
            <Filter className="h-3.5 w-3.5" />
            <span>Filter</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="performance" className="space-y-4">
          <TabsList>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="win-loss">Win/Loss Analysis</TabsTrigger>
            <TabsTrigger value="history">Trade History</TabsTrigger>
          </TabsList>
          <TabsContent value="performance" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Trades</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">245</div>
                  <p className="text-xs text-muted-foreground">Last 30 days</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">68.2%</div>
                  <p className="text-xs text-muted-foreground">167 winning trades</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Average Profit</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-500">+$125.45</div>
                  <p className="text-xs text-muted-foreground">Per winning trade</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Average Loss</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-red-500">-$52.30</div>
                  <p className="text-xs text-muted-foreground">Per losing trade</p>
                </CardContent>
              </Card>
            </div>
            <Card>
              <CardHeader>
                <CardTitle>Performance Over Time</CardTitle>
              </CardHeader>
              <CardContent>
                <PerformanceChart />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="win-loss" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Win/Loss Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <WinLossChart />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Trade Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">Win Rate</h4>
                        <p className="text-2xl font-bold">68.2%</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">Loss Rate</h4>
                        <p className="text-2xl font-bold">31.8%</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">Profit Factor</h4>
                        <p className="text-2xl font-bold">2.4</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">Risk/Reward</h4>
                        <p className="text-2xl font-bold">1:2.3</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">Max Drawdown</h4>
                        <p className="text-2xl font-bold">12.5%</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">Avg Trade Duration</h4>
                        <p className="text-2xl font-bold">3h 15m</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="history">
            <TradeHistoryTable />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

