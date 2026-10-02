import { expect, test } from '@playwright/test'
import { ANSWER_EVENTS, mockApi } from './mock-api'

const EXAMPLE = 'How did Infosys revenue and operating margin change in Q1 FY27?'

test('asks a question and shows the streamed, cited, calculated answer', async ({ page }) => {
  const { asks } = await mockApi(page)
  await page.goto('/')

  await expect(page.getByRole('heading', { name: /Ask the annual reports/ })).toBeVisible()
  await expect(page.getByText('2 filings · 3 companies')).toBeVisible()

  await page.getByRole('button', { name: EXAMPLE }).click()

  await expect(page.getByRole('heading', { name: EXAMPLE })).toBeVisible()
  await expect(page.getByText(/Revenue was \$5,082 million/)).toBeVisible()
  await expect(page.getByText('up 2.85% year over year')).toBeVisible()
  await expect(page.getByText('Filtered to')).toBeVisible()
  await expect(page.getByText('(5082-4941)/4941*100')).toBeVisible() // calculator chip
  await expect(page.getByText('2.853674')).toBeVisible()
  await expect(page.getByText('2 passages retrieved · 2 cited')).toBeVisible()
  expect(asks).toHaveLength(1)
  expect(asks[0]).toEqual({ question: EXAMPLE, history: [] })
})

test('clicking a citation reveals the passage with its retrieval scores, and p. N opens the PDF page', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')
  await page.getByRole('button', { name: EXAMPLE }).click()
  await expect(page.getByText(/Revenue was \$5,082 million/)).toBeVisible()

  // Scores are only shown for the active passage.
  await expect(page.getByText('cos')).toHaveCount(0)
  await page.locator('.answer').getByRole('button', { name: '2', exact: true }).click()
  for (const badge of [/^cos 0\.810$/, /^bm25 12\.40$/, /^rrf 0\.0328$/, /^rerank 7\.50$/, /^ranks d1 \/ k2$/]) {
    await expect(page.getByText(badge)).toBeVisible()
  }

  await page.getByRole('button', { name: 'p. 5' }).click()
  const viewer = page.locator('iframe[title="Source document"]')
  await expect(viewer).toHaveAttribute('src', /\/api\/documents\/infosys-q1\/pdf#page=5$/)
  await expect(page.getByText('infosys-q1 · page 5')).toBeVisible()

  await page.getByTitle('Back to sources').click()
  await expect(viewer).toHaveCount(0)
  await expect(page.getByText('2 passages retrieved')).toBeVisible()
})

test('a follow-up sends the earlier turn as history and shows the rewritten question', async ({ page }) => {
  const rewritten = 'What was Wipro revenue in Q1 FY2027?'
  const { asks } = await mockApi(page, {
    onAsk: (body) =>
      body.history.length === 0
        ? ANSWER_EVENTS
        : [{ type: 'rewrite', original: body.question, question: rewritten }, ...ANSWER_EVENTS],
  })
  await page.goto('/')
  await page.getByRole('button', { name: EXAMPLE }).click()
  await expect(page.getByText('up 2.85% year over year')).toBeVisible()

  const box = page.getByPlaceholder(/Ask a follow-up/)
  await box.fill('and Wipro?')
  await box.press('Enter')

  await expect(page.getByText(rewritten)).toBeVisible()
  expect(asks).toHaveLength(2)
  expect(asks[1].question).toBe('and Wipro?')
  expect(asks[1].history).toHaveLength(1)
  expect(asks[1].history[0].question).toBe(EXAMPLE)
  expect(asks[1].history[0].answer).toContain('Revenue was $5,082 million [1]')

  // The next follow-up carries the rewritten (standalone) form of the previous one.
  await box.fill('what about margins?')
  await box.press('Enter')
  await expect.poll(() => asks.length).toBe(3)
  expect(asks[2].history.map((h) => h.question)).toEqual([EXAMPLE, rewritten])
})

test('shows a refusal without sources when the answer is not in the filings', async ({ page }) => {
  await mockApi(page, {
    onAsk: () => [
      { type: 'analysis', companies: [], doc_type: null, comparison: false, best_similarity: 0.3, retrieval_ms: 20 },
      { type: 'sources', sources: [] },
      { type: 'token', text: "I couldn't find this in the indexed filings." },
      { type: 'done', answered: false, cited: [], gated: true, total_ms: 40 },
    ],
  })
  await page.goto('/')
  const box = page.getByPlaceholder(/Ask about revenue/)
  await box.fill('Who won the football world cup?')
  await box.press('Enter')

  await expect(page.getByText("I couldn't find this in the indexed filings.")).toBeVisible()
  await expect(page.getByText('All companies')).toBeVisible()
  await expect(page.getByText(/The passages retrieved for your question appear here/)).toBeVisible()
})

test('shows the API error message (rate limit) in the conversation', async ({ page }) => {
  await mockApi(page, { onAsk: () => ({ status: 429, detail: 'Rate limit: 30 questions per hour. Try again in 12 min.' }) })
  await page.goto('/')
  await page.getByRole('button', { name: EXAMPLE }).click()
  await expect(page.getByText('Rate limit: 30 questions per hour. Try again in 12 min.')).toBeVisible()
})

test('New chat clears the conversation', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')
  await page.getByRole('button', { name: EXAMPLE }).click()
  await expect(page.getByText('up 2.85% year over year')).toBeVisible()

  await page.getByTitle('Start a new conversation').click()
  await expect(page.getByRole('heading', { name: /Ask the annual reports/ })).toBeVisible()
  await expect(page.getByText('up 2.85% year over year')).toHaveCount(0)
})

test('Library and Evaluation tabs render and are deep-linkable', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')

  await page.getByRole('button', { name: 'Library' }).click()
  await expect(page.getByText(/2 documents · 22 pages/)).toBeVisible()
  await expect(page.getByText('Infosys Q1 FY27 earnings')).toBeVisible()
  expect(page.url()).toMatch(/#library$/)

  await page.getByRole('button', { name: 'Evaluation' }).click()
  await expect(page.getByText('Hybrid + filters + rerank')).toBeVisible()
  await expect(page.getByText('96.1%', { exact: true })).toBeVisible()

  await page.goto('/#eval')
  await expect(page.getByText('BM25 only')).toBeVisible()
})

test('tells the user when the API is unreachable', async ({ page }) => {
  await mockApi(page, { documentsFail: true })
  await page.goto('/')
  await expect(page.getByText(/Can't reach the API/)).toBeVisible()
  await expect(page.getByRole('button', { name: 'Retry' })).toBeVisible()
})
