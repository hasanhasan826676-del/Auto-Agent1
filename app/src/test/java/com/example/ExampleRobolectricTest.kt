package com.example

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.example.agent.AgentActionType
import com.example.ai.GeminiAiPlanner
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class ExampleRobolectricTest {

    @Test
    fun `read app name string from context`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val appName = context.getString(R.string.app_name)
        assertEquals("TapPilot Agent", appName)
    }

    @Test
    fun `gemini planner parses structured JSON action with waitSeconds and memoryNote`() {
        val planner = GeminiAiPlanner()
        val json = """
            {
              "actionType": "WAIT",
              "targetNodeId": null,
              "x": null,
              "y": null,
              "textToType": null,
              "waitSeconds": 45,
              "memoryNote": "Posted in group; awaiting Messenger reply.",
              "reason": "Waiting for seller reply",
              "requiresConfirmation": false
            }
        """.trimIndent()

        val parsed = planner.parsePlannedAction(json)
        assertEquals(AgentActionType.WAIT, parsed.actionType)
        assertEquals(45, parsed.waitSeconds)
        assertEquals("Posted in group; awaiting Messenger reply.", parsed.memoryNote)
        assertFalse(parsed.requiresConfirmation)
    }

    @Test
    fun `gemini planner parses DRAG shot with learnedRule and foulOrMistakeLesson`() {
        val planner = GeminiAiPlanner()
        val json = """
            {
              "actionType": "DRAG",
              "targetNodeId": null,
              "x": 540,
              "y": 1620,
              "endX": 460,
              "endY": 1820,
              "durationMs": 420,
              "textToType": null,
              "waitSeconds": null,
              "memoryNote": "Aiming striker for top-right white coin",
              "learnedRule": "Pull striker slightly left-down to sink top-right pocket coin.",
              "foulOrMistakeLesson": "Avoid pulling straight down when center coin is near pocket edge.",
              "reason": "Pull striker opposite target vector to pocket white coin",
              "requiresConfirmation": false
            }
        """.trimIndent()

        val parsed = planner.parsePlannedAction(json)
        assertEquals(AgentActionType.DRAG, parsed.actionType)
        assertEquals(540, parsed.x)
        assertEquals(1620, parsed.y)
        assertEquals(460, parsed.endX)
        assertEquals(1820, parsed.endY)
        assertEquals(420L, parsed.durationMs)
        assertEquals("Pull striker slightly left-down to sink top-right pocket coin.", parsed.learnedRule)
        assertEquals(
            "Avoid pulling straight down when center coin is near pocket edge.",
            parsed.foulOrMistakeLesson
        )
        assertFalse(parsed.requiresConfirmation)
    }

    @Test
    fun `research playbook parses and serializes cleanly`() {
        val planner = GeminiAiPlanner()
        val json = """
            {
              "topicKey": "carrom_board_game",
              "topicTitle": "Carrom Board Game",
              "rulesSummary": "Position striker on baseline and drag opposite to shoot coins into 4 corner pockets.",
              "winningStrategies": [
                "Align striker center opposite target coin center toward corner pocket."
              ],
              "foulLessons": [
                "Never pocket the striker alone; causes foul and returns one pocketed coin."
              ]
            }
        """.trimIndent()

        val pb = planner.parseResearchPlaybook(json, "Play Carrom game")
        val encoded = com.example.agent.AgentKnowledgeStore.serializePlaybooksJson(listOf(pb))
        val decoded = com.example.agent.AgentKnowledgeStore.parsePlaybooksJson(encoded)

        assertEquals(1, decoded.size)
        assertEquals("carrom_board_game", decoded[0].topicKey)
        assertEquals("Carrom Board Game", decoded[0].topicTitle)
        assertEquals(1, decoded[0].winningStrategies.size)
        assertEquals(1, decoded[0].foulLessons.size)
    }

    @Test
    fun `action cache serializes and parses entries cleanly`() {
        val entry = com.example.agent.CachedActionEntry(
            cacheKey = "abc1234567",
            goalKeywords = "carrom_classic_play",
            packageName = "com.miniclip.carrom",
            screenSignature = "sig999",
            actionType = AgentActionType.TAP,
            targetLabel = "Play Classic",
            targetViewId = "btn_play",
            tapX = 190,
            tapY = 340,
            textToType = null,
            hitCount = 3,
            lastUsedMs = 123456789L
        )
        val encoded = com.example.agent.ActionCacheStore.serializeJson(listOf(entry))
        val decoded = com.example.agent.ActionCacheStore.parseJson(encoded)

        assertEquals(1, decoded.size)
        assertEquals("btn_play", decoded[0].targetViewId)
        assertEquals(3, decoded[0].hitCount)
        assertEquals(190, decoded[0].tapX)
    }

    @Test
    fun `visual grid annotator preserves dimensions without mutating source bitmap`() {
        val source = android.graphics.Bitmap.createBitmap(540, 1200, android.graphics.Bitmap.Config.ARGB_8888)
        source.eraseColor(android.graphics.Color.rgb(210, 180, 140))

        val annotated = com.example.ai.VisualGridAnnotator.createAnnotatedCopy(
            source = source,
            screenWidth = 1080,
            screenHeight = 2400
        )

        assertEquals(540, annotated.width)
        assertEquals(1200, annotated.height)
        assertFalse(annotated === source)
        // Corner pixel at (0,0) of immutable source remains intact while annotated copy draws grid lines
        assertEquals(android.graphics.Color.rgb(210, 180, 140), source.getPixel(108, 600))
        annotated.recycle()
        source.recycle()
    }

    @Test
    fun `carrom vision detector finds painted discs and solves shot offline without AI calls`() {
        com.example.game.CarromVisionDetector.resetMotionHistory()
        val width = 540
        val height = 1000
        val bm = android.graphics.Bitmap.createBitmap(width, height, android.graphics.Bitmap.Config.ARGB_8888)
        // Warm tan carrom wooden background (neither white, black, red queen, nor cyan striker)
        bm.eraseColor(android.graphics.Color.rgb(205, 165, 110))

        fun paintDisc(cx: Float, cy: Float, r: Float, colorInt: Int) {
            val rSq = r * r
            val minX = (cx - r).toInt().coerceIn(0, width - 1)
            val maxX = (cx + r).toInt().coerceIn(0, width - 1)
            val minY = (cy - r).toInt().coerceIn(0, height - 1)
            val maxY = (cy + r).toInt().coerceIn(0, height - 1)
            for (y in minY..maxY) {
                for (x in minX..maxX) {
                    val dx = x - cx
                    val dy = y - cy
                    if (dx * dx + dy * dy <= rSq) {
                        bm.setPixel(x, y, colorInt)
                    }
                }
            }
        }

        val boardLeft = width * 0.06f
        val boardWidth = width * 0.88f
        val boardTop = (height - boardWidth) * 0.48f
        val baselineY = boardTop + boardWidth * 0.855f

        // Draw target white coin aimed comfortably toward top-left corner
        paintDisc(boardLeft + boardWidth * 0.34f, boardTop + boardWidth * 0.36f, 15f, android.graphics.Color.rgb(248, 248, 245))
        // Draw Queen near center
        paintDisc(boardLeft + boardWidth * 0.50f, boardTop + boardWidth * 0.50f, 15f, android.graphics.Color.rgb(225, 28, 35))
        // Draw cyan striker on baseline
        paintDisc(boardLeft + boardWidth * 0.58f, baselineY, 18f, android.graphics.Color.rgb(0, 215, 235))

        val outcome = com.example.game.CarromVisionDetector.analyzeAndSolve(
            screenshot = bm,
            screenWidth = 1080,
            screenHeight = 2000,
            myCoinColor = com.example.game.CoinColor.WHITE,
            consecutiveOfflineWaits = 0
        )

        org.junit.Assert.assertTrue(
            "Expected SolvedShot from local vision detector but got $outcome",
            outcome is com.example.game.CarromVisionOutcome.SolvedShot
        )

        // Sudden large visual change on next frame triggers offline stillness wait (0 API)
        bm.eraseColor(android.graphics.Color.rgb(30, 30, 30))
        val motionOutcome = com.example.game.CarromVisionDetector.analyzeAndSolve(
            screenshot = bm,
            screenWidth = 1080,
            screenHeight = 2000,
            myCoinColor = com.example.game.CoinColor.WHITE,
            consecutiveOfflineWaits = 0
        )
        org.junit.Assert.assertTrue(motionOutcome is com.example.game.CarromVisionOutcome.WaitStillnessOrTurn)
        bm.recycle()
        com.example.game.CarromVisionDetector.resetMotionHistory()
    }

    @Test
    fun `carrom vision detector anchors non-cyan ivory striker using bottom slider knob`() {
        com.example.game.CarromVisionDetector.resetMotionHistory()
        val width = 540
        val height = 1000
        val bm = android.graphics.Bitmap.createBitmap(width, height, android.graphics.Bitmap.Config.ARGB_8888)
        bm.eraseColor(android.graphics.Color.rgb(212, 174, 122))

        fun paintDisc(cx: Float, cy: Float, r: Float, colorInt: Int) {
            val rSq = r * r
            val minX = (cx - r).toInt().coerceIn(0, width - 1)
            val maxX = (cx + r).toInt().coerceIn(0, width - 1)
            val minY = (cy - r).toInt().coerceIn(0, height - 1)
            val maxY = (cy + r).toInt().coerceIn(0, height - 1)
            for (y in minY..maxY) {
                for (x in minX..maxX) {
                    val dx = x - cx
                    val dy = y - cy
                    if (dx * dx + dy * dy <= rSq) {
                        bm.setPixel(x, y, colorInt)
                    }
                }
            }
        }

        val boardLeft = width * 0.055f
        val boardWidth = width * 0.89f
        val boardTop = (height - boardWidth) * 0.48f
        val boardBottom = boardTop + boardWidth
        val baselineY = boardTop + boardWidth * 0.845f
        val strikerX = boardLeft + boardWidth * 0.50f

        // Target white coin + ivory/white striker (not cyan!) + bottom gold slider knob below board
        paintDisc(boardLeft + boardWidth * 0.30f, boardTop + boardWidth * 0.35f, 15f, android.graphics.Color.rgb(248, 248, 245))
        paintDisc(strikerX, baselineY, 19f, android.graphics.Color.rgb(245, 244, 238))
        paintDisc(strikerX, boardBottom + height * 0.06f, 16f, android.graphics.Color.rgb(255, 200, 30))

        val outcome = com.example.game.CarromVisionDetector.analyzeAndSolve(
            screenshot = bm,
            screenWidth = 1080,
            screenHeight = 2000,
            myCoinColor = com.example.game.CoinColor.WHITE,
            consecutiveOfflineWaits = 0
        )

        org.junit.Assert.assertTrue(
            "Expected SolvedShot when ivory striker + bottom slider knob are present, got $outcome",
            outcome is com.example.game.CarromVisionOutcome.SolvedShot
        )
        val solved = outcome as com.example.game.CarromVisionOutcome.SolvedShot
        org.junit.Assert.assertNotNull(solved.detectedStriker)
        org.junit.Assert.assertNotNull(solved.detectedSliderKnob)

        bm.recycle()
        com.example.game.CarromVisionDetector.resetMotionHistory()
    }

    @Test
    fun `contact profile store serializes and parses platform and relationship json cleanly`() {
        val contacts = listOf(
            com.example.agent.ContactProfile(
                id = "c1",
                name = "Rahim",
                platform = com.example.agent.ChatPlatform.FACEBOOK_MESSENGER,
                relationship = com.example.agent.RelationshipType.FRIEND,
                customStyleNote = "Call him mama",
                autoReplyEnabled = true,
                createdAtMs = 1000L
            ),
            com.example.agent.ContactProfile(
                id = "c2",
                name = "Sara",
                platform = com.example.agent.ChatPlatform.WHATSAPP,
                relationship = com.example.agent.RelationshipType.GF,
                customStyleNote = "Warm sweet tone",
                autoReplyEnabled = true,
                createdAtMs = 2000L
            )
        )
        val encoded = com.example.agent.ContactProfileStore.serializeJson(contacts)
        val decoded = com.example.agent.ContactProfileStore.parseJson(encoded)

        assertEquals(2, decoded.size)
        assertEquals("Rahim", decoded[0].name)
        assertEquals(com.example.agent.ChatPlatform.FACEBOOK_MESSENGER, decoded[0].platform)
        assertEquals(com.example.agent.RelationshipType.FRIEND, decoded[0].relationship)
        assertEquals("Sara", decoded[1].name)
        assertEquals(com.example.agent.ChatPlatform.WHATSAPP, decoded[1].platform)
        assertEquals(com.example.agent.RelationshipType.GF, decoded[1].relationship)
    }

    @Test
    fun `carrom shot memory serializes and parses pocket power and placement stats cleanly`() {
        val initial = com.example.game.CarromMemoryData()
        val shot = com.example.game.PendingCarromShotRecord(
            myCoinColor = com.example.game.CoinColor.WHITE,
            targetCoinColor = com.example.game.CoinColor.WHITE,
            pocketIndex = 1,
            cutAngleDeg = 10f,
            powerStyle = com.example.game.CarromPowerStyle.HARD,
            movedStrikerWithSlider = true,
            isBankShot = false,
            isComboShot = false,
            isBreakShot = false,
            preCounts = com.example.game.BoardPieceCounts(myCount = 5, opponentCount = 5, queenCount = 1)
        )
        val eval = com.example.game.CarromShotMemoryStore.evaluateShotOutcome(
            current = initial,
            pendingShot = shot,
            postCounts = com.example.game.BoardPieceCounts(myCount = 6, opponentCount = 5, queenCount = 1)
        )

        val encoded = com.example.game.CarromShotMemoryStore.serializeJson(eval.updatedMemory)
        val decoded = com.example.game.CarromShotMemoryStore.parseJson(encoded)

        assertEquals(1, decoded.straightFoulCount)
        assertEquals(1, decoded.placementStats.movedAttempts)
        assertEquals(1, decoded.placementStats.movedFouls)
        assertEquals(1, decoded.recentFouls.size)
        org.junit.Assert.assertTrue(decoded.learnedPowerScaleFor(1, 10f) < 1.0f)
        org.junit.Assert.assertTrue(decoded.formatForPrompt().contains("Top-Right Pocket"))
    }
}
