import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Answer } from './Answer'

const noop = () => {}

describe('Answer', () => {
  it('renders citations as buttons that report the source number', async () => {
    const onCite = vi.fn()
    render(<Answer text="Revenue was $5,082 million [2], up 2.85% [1][3]." streaming={false} activeSource={null} onCite={onCite} />)

    await userEvent.click(screen.getByRole('button', { name: '2' }))
    expect(onCite).toHaveBeenCalledWith(2)
    expect(screen.getAllByRole('button').map((b) => b.textContent)).toEqual(['2', '1', '3'])
  })

  it('highlights the active source', () => {
    render(<Answer text="A [1] B [2]" streaming={false} activeSource={2} onCite={noop} />)
    expect(screen.getByRole('button', { name: '2' }).className).toContain('bg-mint-400')
    expect(screen.getByRole('button', { name: '1' }).className).not.toContain('bg-mint-400')
  })

  it('opens ordinary links in a new tab', () => {
    render(<Answer text="See [the filing](https://www.sec.gov)." streaming={false} activeSource={null} onCite={noop} />)
    const link = screen.getByRole('link', { name: 'the filing' })
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noreferrer')
  })

  it('renders GitHub-flavoured tables', () => {
    const table = '| Company | Revenue |\n|---|---|\n| Infosys | 5,082 |'
    render(<Answer text={table} streaming={false} activeSource={null} onCite={noop} />)
    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: '5,082' })).toBeInTheDocument()
  })

  it('shows the streaming caret only while streaming', () => {
    const { container, rerender } = render(<Answer text="Hi" streaming activeSource={null} onCite={noop} />)
    expect(container.firstChild).toHaveClass('caret')
    rerender(<Answer text="Hi" streaming={false} activeSource={null} onCite={noop} />)
    expect(container.firstChild).not.toHaveClass('caret')
  })
})
