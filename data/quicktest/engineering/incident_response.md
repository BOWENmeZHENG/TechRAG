# Incident Response

*Fictional sample document for the TechRAG demo.*

## Severity levels

| Level | Meaning | First response |
| --- | --- | --- |
| SEV1 | Customers cannot pay, or data is exposed | 15 minutes |
| SEV2 | A major feature is degraded | 1 hour |
| SEV3 | A minor issue with a workaround | Next business day |

## Roles

The on-call engineer who is paged becomes incident commander until they hand the role over. For a
SEV1, also name a communications lead, who posts a status update every 30 minutes.

## On-call rotation

Rotations last one week and change on Tuesdays at 10:00 Pacific. If a page is not acknowledged
within 10 minutes, it escalates to the secondary engineer.

## After the incident

SEV1 and SEV2 incidents need a blameless written review within five business days, with an owner
and a due date for every follow-up action.
