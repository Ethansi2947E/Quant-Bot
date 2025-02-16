"use client"

import type React from "react"
import { ArrowUpRight, ArrowDownRight } from "lucide-react"
import { Card } from "@/components/ui/card"

interface Trade {
  symbol: string
  type: string
  entry: number
  sl: number
  tp: number
  profit: number
}

interface ActiveTradesProps {
  data: Trade[]
}

const ActiveTrades: React.FC<ActiveTradesProps> = ({ data }) => {
  if (!data?.length) {
    return (
      <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
        <h3 className="text-xl font-bold text-white mb-4">Active Trades</h3>
        <p className="text-white/60">No active trades at the moment.</p>
      </Card>
    )
  }

  return (
    <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold text-white">Active Trades</h3>
          <p className="text-sm text-white/60 mt-1">Currently open positions</p>
        </div>
        <span className="px-3 py-1.5 rounded-full text-sm font-medium bg-green-500/20 text-green-500">
          {data.length} positions
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-sm text-white/60">
              <th className="pb-4 font-medium">Symbol</th>
              <th className="pb-4 font-medium">Type</th>
              <th className="pb-4 font-medium">Entry</th>
              <th className="pb-4 font-medium">Stop Loss</th>
              <th className="pb-4 font-medium">Take Profit</th>
              <th className="pb-4 font-medium">Current Profit</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {data.map((trade, index) => (
              <tr key={index}>
                <td className="py-4 font-medium text-white">{trade.symbol}</td>
                <td className="py-4">
                  <span
                    className={`px-2 py-1 rounded-full text-xs font-medium ${
                      trade.type === "BUY" ? "bg-green-500/20 text-green-500" : "bg-red-500/20 text-red-500"
                    }`}
                  >
                    {trade.type}
                  </span>
                </td>
                <td className="py-4 text-white">{trade.entry.toFixed(5)}</td>
                <td className="py-4 text-white">{trade.sl.toFixed(5)}</td>
                <td className="py-4 text-white">{trade.tp.toFixed(5)}</td>
                <td className="py-4">
                  <div className={`flex items-center gap-1 ${trade.profit >= 0 ? "text-green-500" : "text-red-500"}`}>
                    {trade.profit >= 0 ? <ArrowUpRight className="h-4 w-4" /> : <ArrowDownRight className="h-4 w-4" />}$
                    {Math.abs(trade.profit).toFixed(2)}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

export default ActiveTrades

