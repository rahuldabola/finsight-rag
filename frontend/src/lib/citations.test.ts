import { describe, expect, it } from 'vitest'
import { citationNumber, linkCitations } from './citations'

describe('linkCitations', () => {
  it('turns [n] into a citation link, including adjacent citations', () => {
    expect(linkCitations('Revenue was $5,082 million [2].')).toBe('Revenue was $5,082 million [2](#cite-2).')
    expect(linkCitations('Both agree [1][4]')).toBe('Both agree [1](#cite-1)[4](#cite-4)')
  })

  it('leaves existing links and non-numeric brackets alone', () => {
    expect(linkCitations('[3](https://example.com) and [note] and [123]')).toBe('[3](https://example.com) and [note] and [123]')
  })
})

describe('citationNumber', () => {
  it('extracts the source number from a citation href', () => {
    expect(citationNumber('#cite-7')).toBe(7)
  })

  it('returns null for ordinary links', () => {
    expect(citationNumber('https://sec.gov')).toBeNull()
    expect(citationNumber(undefined)).toBeNull()
  })
})
