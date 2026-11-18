package com.example.game

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs

sealed interface CarromVisionOutcome {
    data class WaitStillnessOrTurn(
        val reason: String,
        val boardPieces: List<CarromPiece> = emptyList()
    ) : CarromVisionOutcome
    data class SolvedShot(
        val solution: CarromShotSolution,
        val detectedStriker: CarromPiece?,
        val board: CarromBoardLayout,
        val detectedSliderKnob: Point2D? = null,
        val boardPieces: List<CarromPiece> = emptyList()
    ) : CarromVisionOutcome
    data class NeedsAiVision(
        val diagnostic: String,
        val boardPieces: List<CarromPiece> = emptyList()
    ) : CarromVisionOutcome
}

/**
 * Pure Kotlin on-device Carrom screenshot analyzer + offline board-motion detector.
 * Zero native OpenCV bloat (~15-35ms per frame). Falls back cleanly to AI Vision
 * whenever board confidence is low or screen is a lobby/menu.
 */
object CarromVisionDetector {

    private const val GRID_SIZE = 16
    private const val SAMPLE_STEP = 4
    private const val MOTION_THRESHOLD = 13.5f

    private var previousBoardLuma: IntArray? = null

    fun resetMotionHistory() {
        previousBoardLuma = null
    }

    fun analyzeAndSolve(
        screenshot: Bitmap?,
        screenWidth: Int,
        screenHeight: Int,
        myCoinColor: CoinColor,
        consecutiveOfflineWaits: Int,
        shotMemory: CarromMemoryData? = null
    ): CarromVisionOutcome {
        if (screenshot == null || screenshot.width < 120 || screenshot.height < 120) {
            return CarromVisionOutcome.NeedsAiVision("No screenshot available for Carrom solver.")
        }

        val scaleX = screenWidth.toFloat() / screenshot.width.toFloat()
        val scaleY = screenHeight.toFloat() / screenshot.height.toFloat()

        // Estimate square Carrom board region (supports portrait & landscape + slight vertical variance)
        val (boardBmLeft, boardBmTop, boardBmRight, boardBmBottom) = estimateBoardBitmapBounds(screenshot)
        val boardBmWidth = (boardBmRight - boardBmLeft).coerceAtLeast(40)
        val boardBmHeight = (boardBmBottom - boardBmTop).coerceAtLeast(40)

        // 1. Offline Motion / Stillness check on inner 80% of board (ignores pulsing edge timers/avatars)
        val innerPadX = (boardBmWidth * 0.10f).toInt()
        val innerPadY = (boardBmHeight * 0.10f).toInt()
        val currentLuma = sampleBoardLumaGrid(
            bitmap = screenshot,
            left = (boardBmLeft + innerPadX).coerceIn(0, screenshot.width - 2),
            top = (boardBmTop + innerPadY).coerceIn(0, screenshot.height - 2),
            right = (boardBmRight - innerPadX).coerceIn(1, screenshot.width - 1),
            bottom = (boardBmBottom - innerPadY).coerceIn(1, screenshot.height - 1)
        )
        val prevLuma = previousBoardLuma
        previousBoardLuma = currentLuma

        if (prevLuma != null && consecutiveOfflineWaits < 2) {
            val meanDiff = computeMeanAbsDiff(prevLuma, currentLuma)
            if (meanDiff > MOTION_THRESHOLD) {
                return CarromVisionOutcome.WaitStillnessOrTurn(
                    "Board motion detected (diff=${meanDiff.toInt()}) — waiting offline for coins to settle (0 API)."
                )
            }
        }

        // 2. Detect bottom slider knob (present below the board when it's our turn in Carrom Pool / King)
        val sliderKnobPx = detectBottomSliderKnob(
            bitmap = screenshot,
            boardLeft = boardBmLeft,
            boardRight = boardBmRight,
            boardBottom = boardBmBottom,
            scaleX = scaleX,
            scaleY = scaleY
        )

        // 3. Scan board region into clustered pieces
        val rawPieces = detectBoardPieces(
            bitmap = screenshot,
            left = boardBmLeft,
            top = boardBmTop,
            right = boardBmRight,
            bottom = boardBmBottom,
            scaleX = scaleX,
            scaleY = scaleY
        )

        val boardLeft = boardBmLeft * scaleX
        val boardTop = boardBmTop * scaleY
        val boardRight = boardBmRight * scaleX
        val boardBottom = boardBmBottom * scaleY
        val boardWidthPx = (boardRight - boardLeft).coerceAtLeast(100f)
        val boardHeightPx = (boardBottom - boardTop).coerceAtLeast(100f)

        val coinRadius = boardWidthPx * 0.029f
        val strikerRadius = boardWidthPx * 0.036f
        val pocketRadius = boardWidthPx * 0.052f
        val pocketInset = boardWidthPx * 0.072f

        val pockets = listOf(
            Point2D(boardLeft + pocketInset, boardTop + pocketInset),
            Point2D(boardRight - pocketInset, boardTop + pocketInset),
            Point2D(boardLeft + pocketInset, boardBottom - pocketInset),
            Point2D(boardRight - pocketInset, boardBottom - pocketInset)
        )
        val defaultBaselineY = boardTop + boardHeightPx * 0.845f

        val promoted = promoteOrAnchorStriker(
            pieces = rawPieces,
            baselineY = defaultBaselineY,
            boardLeft = boardLeft,
            boardRight = boardRight,
            boardHeightPx = boardHeightPx,
            strikerRadiusPx = strikerRadius,
            sliderKnobPx = sliderKnobPx
        )

        if (promoted.size < 2) {
            return CarromVisionOutcome.NeedsAiVision(
                diagnostic = "Detected ${promoted.size} Carrom piece(s); delegating to AI Vision.",
                boardPieces = promoted
            )
        }

        val strikerPiece = promoted.firstOrNull { it.color == CoinColor.STRIKER }
        if (strikerPiece == null && sliderKnobPx == null) {
            return if (consecutiveOfflineWaits < 1) {
                // Wait at most 1 short cycle if baseline looks empty (opponent turn), then let AI inspect.
                CarromVisionOutcome.WaitStillnessOrTurn(
                    reason = "Striker not on baseline (likely opponent turn) — waiting offline (0 API).",
                    boardPieces = promoted
                )
            } else {
                CarromVisionOutcome.NeedsAiVision(
                    diagnostic = "Striker not matched locally after wait; delegating to AI Vision.",
                    boardPieces = promoted
                )
            }
        }

        val calibratedBaselineY = strikerPiece?.center?.y ?: defaultBaselineY
        val boardLayout = CarromBoardLayout(
            left = boardLeft,
            top = boardTop,
            right = boardRight,
            bottom = boardBottom,
            pockets = pockets,
            baselineY = calibratedBaselineY,
            coinRadiusPx = coinRadius,
            strikerRadiusPx = strikerRadius,
            pocketRadiusPx = pocketRadius
        )

        val solution = CarromGeometrySolver.solveBestShot(
            board = boardLayout,
            pieces = promoted,
            myCoinColor = myCoinColor,
            screenWidth = screenWidth,
            screenHeight = screenHeight,
            existingStriker = strikerPiece,
            canRepositionStriker = sliderKnobPx != null,
            shotMemory = shotMemory
        )

        return if (solution != null && solution.confidenceScore >= 0.50f) {
            CarromVisionOutcome.SolvedShot(
                solution = solution,
                detectedStriker = strikerPiece,
                board = boardLayout,
                detectedSliderKnob = sliderKnobPx,
                boardPieces = promoted
            )
        } else {
            CarromVisionOutcome.NeedsAiVision(
                diagnostic = "No high-confidence offline Carrom shot found; using AI Vision.",
                boardPieces = promoted
            )
        }
    }

    fun inspectBoardCounts(
        screenshot: Bitmap?,
        screenWidth: Int,
        screenHeight: Int,
        myCoinColor: CoinColor
    ): BoardPieceCounts? {
        if (screenshot == null || screenshot.width < 120 || screenshot.height < 120) return null
        val scaleX = screenWidth.toFloat() / screenshot.width.toFloat()
        val scaleY = screenHeight.toFloat() / screenshot.height.toFloat()
        val bounds = estimateBoardBitmapBounds(screenshot)
        val raw = detectBoardPieces(
            bitmap = screenshot,
            left = bounds.left,
            top = bounds.top,
            right = bounds.right,
            bottom = bounds.bottom,
            scaleX = scaleX,
            scaleY = scaleY
        )
        if (raw.isEmpty()) return null
        return CarromShotMemoryStore.countBoardPieces(raw, myCoinColor)
    }

    private data class BoardBmBounds(
        val left: Int,
        val top: Int,
        val right: Int,
        val bottom: Int
    )

    private fun estimateBoardBitmapBounds(bitmap: Bitmap): BoardBmBounds {
        val w = bitmap.width
        val h = bitmap.height
        val isLandscape = w > h

        if (isLandscape) {
            val side = (h * 0.84f).toInt().coerceAtLeast(40)
            val left = ((w - side) / 2).coerceIn(0, w - 40)
            val top = ((h - side) / 2).coerceIn(0, h - 40)
            return BoardBmBounds(
                left = left,
                top = top,
                right = (left + side).coerceAtMost(w - 1),
                bottom = (top + side).coerceAtMost(h - 1)
            )
        }

        val left = (w * 0.055f).toInt()
        val right = (w * 0.945f).toInt()
        val boardW = (right - left).coerceAtLeast(40)
        val defaultTop = ((h - boardW) * 0.48f).toInt().coerceIn(0, h - 40)

        // Refine top within +-8% of screen height only if there is clear inside-vs-outside border contrast
        val xStart = left + (boardW * 0.25f).toInt()
        val xEnd = right - (boardW * 0.25f).toInt()
        val searchMinY = (defaultTop - (h * 0.08f).toInt()).coerceAtLeast(0)
        val searchMaxY = (defaultTop + (h * 0.08f).toInt()).coerceAtMost(h - boardW - 1).coerceAtLeast(searchMinY)

        fun avgRowLuma(py: Int): Float {
            val clampedY = py.coerceIn(0, h - 1)
            var rowLuma = 0
            var count = 0
            var px = xStart
            while (px <= xEnd) {
                val p = bitmap.getPixel(px, clampedY)
                rowLuma += (Color.red(p) * 30 + Color.green(p) * 59 + Color.blue(p) * 11) / 100
                count++
                px += 12
            }
            return if (count > 0) rowLuma.toFloat() / count else 0f
        }

        fun scoreCandidate(candidateTop: Int): Float {
            var insideSum = 0f
            val probeRows = 6
            for (r in 1..probeRows) {
                val py = candidateTop + (boardW * r) / (probeRows + 1)
                insideSum += avgRowLuma(py)
            }
            val insideAvg = insideSum / probeRows
            val aboveY = (candidateTop - 16).coerceAtLeast(0)
            val belowY = (candidateTop + boardW + 16).coerceAtMost(h - 1)
            val outsideAvg = (avgRowLuma(aboveY) + avgRowLuma(belowY)) * 0.5f
            return insideAvg - outsideAvg
        }

        val defaultScore = scoreCandidate(defaultTop)
        var bestTop = defaultTop
        var bestScore = defaultScore
        var y = searchMinY
        while (y <= searchMaxY) {
            val score = scoreCandidate(y)
            if (score > bestScore) {
                bestScore = score
                bestTop = y
            }
            y += 6
        }

        val finalTop = if (bestScore > defaultScore + 15f && bestScore > 25f) bestTop else defaultTop
        val finalBottom = (finalTop + boardW).coerceAtMost(h - 1)
        return BoardBmBounds(left = left, top = finalTop, right = right, bottom = finalBottom)
    }

    private fun detectBottomSliderKnob(
        bitmap: Bitmap,
        boardLeft: Int,
        boardRight: Int,
        boardBottom: Int,
        scaleX: Float,
        scaleY: Float
    ): Point2D? {
        val sliderTop = (boardBottom + (bitmap.height * 0.015f).toInt()).coerceIn(0, bitmap.height - 2)
        val sliderBottom = (boardBottom + (bitmap.height * 0.165f).toInt()).coerceIn(sliderTop + 1, bitmap.height - 1)
        val bw = (boardRight - boardLeft).coerceAtLeast(1)
        val bh = (sliderBottom - sliderTop).coerceAtLeast(1)
        val cols = (bw / SAMPLE_STEP).coerceAtLeast(1)
        val rows = (bh / SAMPLE_STEP).coerceAtLeast(1)
        if (cols < 4 || rows < 2) return null

        val active = Array(rows) { BooleanArray(cols) }
        val hsv = FloatArray(3)
        for (r in 0 until rows) {
            val py = (sliderTop + r * SAMPLE_STEP).coerceIn(0, bitmap.height - 1)
            for (c in 0 until cols) {
                val px = (boardLeft + c * SAMPLE_STEP).coerceIn(0, bitmap.width - 1)
                rgbToHsv(bitmap.getPixel(px, py), hsv)
                val h = hsv[0]
                val s = hsv[1]
                val v = hsv[2]
                // Slider knobs are vivid glowing discs (gold/orange/cyan/white) — exclude muted wood tones
                active[r][c] = (s >= 0.65f && v >= 0.68f && h in 20f..65f) ||
                    (s >= 0.48f && v >= 0.55f && h in 160f..225f) ||
                    (s <= 0.18f && v >= 0.92f)
            }
        }

        val visited = Array(rows) { BooleanArray(cols) }
        val queue = ArrayDeque<Pair<Int, Int>>()
        var bestCenter: Point2D? = null
        var bestCount = 0

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                if (!active[r][c] || visited[r][c]) continue
                var sumX = 0f
                var sumY = 0f
                var count = 0
                queue.add(r to c)
                visited[r][c] = true

                while (queue.isNotEmpty()) {
                    val (cr, cc) = queue.removeFirst()
                    sumX += (boardLeft + cc * SAMPLE_STEP) * scaleX
                    sumY += (sliderTop + cr * SAMPLE_STEP) * scaleY
                    count++
                    for (dr in -1..1) {
                        for (dc in -1..1) {
                            val nr = cr + dr
                            val nc = cc + dc
                            if (nr in 0 until rows && nc in 0 until cols && !visited[nr][nc] && active[nr][nc]) {
                                visited[nr][nc] = true
                                queue.add(nr to nc)
                            }
                        }
                    }
                }

                // Knob should be a compact disc (not a wide full-width bar)
                if (count in 5..110 && count > bestCount) {
                    bestCount = count
                    bestCenter = Point2D(sumX / count, sumY / count)
                }
            }
        }
        return bestCenter
    }

    private fun promoteOrAnchorStriker(
        pieces: List<CarromPiece>,
        baselineY: Float,
        boardLeft: Float,
        boardRight: Float,
        boardHeightPx: Float,
        strikerRadiusPx: Float,
        sliderKnobPx: Point2D?
    ): List<CarromPiece> {
        if (pieces.any { it.color == CoinColor.STRIKER }) return pieces

        val baselineTolerance = boardHeightPx * 0.08f
        val baselineCandidates = pieces.filter { abs(it.center.y - baselineY) <= baselineTolerance }

        // 1. If bottom slider knob is detected, match the piece on the baseline closest to slider X
        if (sliderKnobPx != null) {
            val maxHorizDelta = (boardRight - boardLeft) * 0.16f
            val matchedBySlider = baselineCandidates.minByOrNull { abs(it.center.x - sliderKnobPx.x) }
            if (matchedBySlider != null && abs(matchedBySlider.center.x - sliderKnobPx.x) <= maxHorizDelta) {
                return pieces.map { piece ->
                    if (piece === matchedBySlider) piece.copy(color = CoinColor.STRIKER) else piece
                }
            }
            // Even if the striker has an uncommon dark/patterned skin, its position matches sliderKnobPx.x on baselineY!
            val clampedX = sliderKnobPx.x.coerceIn(
                boardLeft + strikerRadiusPx * 2.1f,
                boardRight - strikerRadiusPx * 2.1f
            )
            return pieces + CarromPiece(
                color = CoinColor.STRIKER,
                center = Point2D(clampedX, baselineY),
                radiusPx = strikerRadiusPx
            )
        }

        // 2. Fallback without slider: promote a larger white/distinct disc sitting squarely on the baseline
        val avgRadius = pieces.map { it.radiusPx }.average().toFloat().coerceAtLeast(14f)
        val tightBaselineCandidate = baselineCandidates
            .filter { abs(it.center.y - baselineY) <= boardHeightPx * 0.048f }
            .maxByOrNull { it.radiusPx }
        if (tightBaselineCandidate != null &&
            (tightBaselineCandidate.radiusPx >= avgRadius * 1.06f || baselineCandidates.size == 1)
        ) {
            return pieces.map { piece ->
                if (piece === tightBaselineCandidate) piece.copy(color = CoinColor.STRIKER) else piece
            }
        }

        return pieces
    }

    private fun detectBoardPieces(
        bitmap: Bitmap,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int,
        scaleX: Float,
        scaleY: Float
    ): List<CarromPiece> {
        val bw = (right - left).coerceAtLeast(1)
        val bh = (bottom - top).coerceAtLeast(1)
        val cols = (bw / SAMPLE_STEP).coerceAtLeast(1)
        val rows = (bh / SAMPLE_STEP).coerceAtLeast(1)

        val labels = Array(rows) { arrayOfNulls<CoinColor>(cols) }
        val hsv = FloatArray(3)
        val pocketMargin = (bw * 0.11f).toInt()

        for (r in 0 until rows) {
            val py = (top + r * SAMPLE_STEP).coerceIn(0, bitmap.height - 1)
            for (c in 0 until cols) {
                val px = (left + c * SAMPLE_STEP).coerceIn(0, bitmap.width - 1)

                // Skip dark corner pocket holes
                val dx = px - left
                val dy = py - top
                val nearCorner = (dx < pocketMargin || dx > bw - pocketMargin) &&
                    (dy < pocketMargin || dy > bh - pocketMargin)
                if (nearCorner) continue

                val pixel = bitmap.getPixel(px, py)
                rgbToHsv(pixel, hsv)
                val h = hsv[0]
                val s = hsv[1]
                val v = hsv[2]

                val isBaselineBand = abs(dy - bh * 0.845f) < bh * 0.085f

                labels[r][c] = when {
                    // Striker: distinct colored disc/ring on baseline (cyan/blue, vivid gold, purple, green)
                    isBaselineBand && (
                        (h in 165f..245f && s >= 0.36f && v >= 0.42f) ||
                            (h in 36f..60f && s >= 0.58f && v >= 0.72f) ||
                            (h in 260f..325f && s >= 0.42f && v >= 0.42f) ||
                            (h in 85f..155f && s >= 0.45f && v >= 0.45f)
                        ) -> CoinColor.STRIKER
                    // Queen: saturated red/crimson
                    (h >= 338f || h <= 16f) && s >= 0.48f && v >= 0.35f -> CoinColor.QUEEN
                    // White coin: bright ivory/white with low saturation
                    s <= 0.25f && v >= 0.74f -> CoinColor.WHITE
                    // Black coin: dark grey/charcoal disc inside wooden board
                    v in 0.06f..0.29f && s <= 0.42f -> CoinColor.BLACK
                    else -> null
                }
            }
        }

        val visited = Array(rows) { BooleanArray(cols) }
        val detected = mutableListOf<CarromPiece>()
        val queue = ArrayDeque<Pair<Int, Int>>()

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val color = labels[r][c] ?: continue
                if (visited[r][c]) continue

                var sumX = 0f
                var sumY = 0f
                var count = 0
                queue.add(r to c)
                visited[r][c] = true

                while (queue.isNotEmpty()) {
                    val (cr, cc) = queue.removeFirst()
                    sumX += (left + cc * SAMPLE_STEP) * scaleX
                    sumY += (top + cr * SAMPLE_STEP) * scaleY
                    count++

                    for (dr in -1..1) {
                        for (dc in -1..1) {
                            if (dr == 0 && dc == 0) continue
                            val nr = cr + dr
                            val nc = cc + dc
                            if (nr in 0 until rows && nc in 0 until cols &&
                                !visited[nr][nc] && labels[nr][nc] == color
                            ) {
                                visited[nr][nc] = true
                                queue.add(nr to nc)
                            }
                        }
                    }
                }

                val maxCluster = if (color == CoinColor.STRIKER) 170 else 110
                if (count in 4..maxCluster) {
                    val cx = sumX / count
                    val cy = sumY / count
                    val radiusApprox = (kotlin.math.sqrt(count.toFloat() / 3.14f) * SAMPLE_STEP * scaleX)
                        .coerceIn(14f, 68f)
                    detected.add(CarromPiece(color, Point2D(cx, cy), radiusApprox))
                }
            }
        }

        return detected
    }

    private fun sampleBoardLumaGrid(
        bitmap: Bitmap,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int
    ): IntArray {
        val out = IntArray(GRID_SIZE * GRID_SIZE)
        val w = (right - left).coerceAtLeast(1)
        val h = (bottom - top).coerceAtLeast(1)
        var idx = 0
        for (gy in 0 until GRID_SIZE) {
            val py = (top + (gy + 0.5f) * h / GRID_SIZE).toInt().coerceIn(0, bitmap.height - 1)
            for (gx in 0 until GRID_SIZE) {
                val px = (left + (gx + 0.5f) * w / GRID_SIZE).toInt().coerceIn(0, bitmap.width - 1)
                val p = bitmap.getPixel(px, py)
                val luma = (Color.red(p) * 30 + Color.green(p) * 59 + Color.blue(p) * 11) / 100
                out[idx++] = luma
            }
        }
        return out
    }

    private fun computeMeanAbsDiff(a: IntArray, b: IntArray): Float {
        if (a.size != b.size || a.isEmpty()) return 0f
        var total = 0L
        for (i in a.indices) {
            total += abs(a[i] - b[i])
        }
        return total.toFloat() / a.size
    }

    private fun rgbToHsv(pixel: Int, out: FloatArray) {
        val r = ((pixel ushr 16) and 0xFF) / 255f
        val g = ((pixel ushr 8) and 0xFF) / 255f
        val b = (pixel and 0xFF) / 255f
        val maxC = maxOf(r, g, b)
        val minC = minOf(r, g, b)
        val delta = maxC - minC

        var hue = when {
            delta < 1e-5f -> 0f
            maxC == r -> 60f * (((g - b) / delta) % 6f)
            maxC == g -> 60f * (((b - r) / delta) + 2f)
            else -> 60f * (((r - g) / delta) + 4f)
        }
        if (hue < 0f) hue += 360f
        val sat = if (maxC < 1e-5f) 0f else delta / maxC

        out[0] = hue
        out[1] = sat
        out[2] = maxC
    }
}
