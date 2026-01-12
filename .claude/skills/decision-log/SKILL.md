---
name: decision-log
description: Document architectural decisions, research findings, and technical choices. Activates when making design decisions, evaluating trade-offs, or recording research outcomes.
allowed-tools: Read, Write, Edit, Glob
---

# Decision Log Skill

All significant decisions must be documented in `docs/decisions/` using the ADR (Architecture Decision Record) format.

## When to Document

- Architecture and design choices
- Technology/library selections
- API integrations chosen
- Data model decisions
- Algorithm/approach selections
- Trade-off resolutions
- Research findings that influence the project

## ADR Format

Each decision file follows this structure:

```markdown
# ADR-XXX: [Title]

**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Deprecated | Superseded
**Deciders**: [Who made this decision]

## Context

What is the issue that we're seeing that motivates this decision?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult because of this change?

## Alternatives Considered

What other options were evaluated?

## References

Links to research, documentation, or related ADRs.
```

## File Naming

- Format: `ADR-XXX-short-title.md`
- Example: `ADR-001-flight-data-api-selection.md`
- Keep titles concise (3-5 words)

## Index File

Maintain `docs/decisions/README.md` as an index of all ADRs with:
- ADR number and title
- Status
- Date
- One-line summary

## Decision Categories

| Category | Prefix | Example |
|----------|--------|---------|
| Architecture | ARCH | ARCH-001-microservices-vs-monolith |
| Data | DATA | DATA-001-database-selection |
| API | API | API-001-flight-data-provider |
| ML | ML | ML-001-forecasting-algorithm |
| Integration | INT | INT-001-weather-api-selection |

## Workflow

1. When facing a decision, create ADR with status "Proposed"
2. Document context, options, and recommendation
3. After team/user approval, change status to "Accepted"
4. Reference ADR in code comments and CLAUDE.md when relevant
