"use client"

import type React from "react"
import { TrendingUp, TrendingDown, Activity } from "lucide-react"
import { Card } from "@/components/ui/card"

interface MarketAnalysisProps {
  data: any
}

const MarketAnalysis: React.FC<MarketAnalysisProps> = ({ data }) => {
  if (!data) return null

  const getTrendIcon = (trend: string) => {
    switch (trend?.toLowerCase()) {
      case "bullish":
        return <TrendingUp className="h-6 w-6 text-green-500" />
      case "bearish":
        return <TrendingDown className="h-6 w-6 text-red-500" />
      default:
        return <Activity className="h-6 w-6 text-blue-500" />
    }
  }

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
        <h3 className="text-xl font-bold text-white mb-6">Market Analysis</h3>
        <div className="flex items-center gap-4 mb-6">
          {getTrendIcon(data.market_bias)}
          <div>
            <p className="text-sm text-white/60">Current Bias</p>
            <p className="text-lg font-semibold text-white">{data.market_bias || "Neutral"}</p>
          </div>
        </div>
        <div>
          <p className="text-sm text-white/60">Structure Type</p>
          <p className="text-lg font-semibold text-white mt-1">{data.structure_type || "Unknown"}</p>
        </div>
      </Card>

      <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
        <h3 className="text-xl font-bold text-white mb-6">Key Levels</h3>
        {data.key_levels?.length > 0 ? (
          <div className="space-y-4">
            {data.key_levels.map((level: any, index: number) => (
              <div key={index} className="flex justify-between items-center">
                <span className="text-white font-medium">{level.price.toFixed(5)}</span>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    level.type === "support" ? "bg-green-500/20 text-green-500" : "bg-red-500/20 text-red-500"
                  }`}
                >
                  {level.type}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-white/60">No key levels identified</p>
        )}
      </Card>
    </div>
  )
}

export default MarketAnalysis

