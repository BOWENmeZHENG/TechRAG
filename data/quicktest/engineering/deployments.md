# Deploying the Payments Service

*Fictional sample document for the TechRAG demo.*

## Release schedule

`payments-api` is released Monday to Thursday between 09:00 and 15:00 Pacific. There are no
releases on Fridays, weekends or company holidays unless the on-call lead approves a hotfix.

## Steps

1. Merge the pull request into `main` after one approving review and a green CI run.
2. The pipeline deploys to staging and runs the smoke tests.
3. Promote the build to production. It receives 5% of traffic for 30 minutes.
4. If the error rate stays below 0.5%, the rollout continues to 100%.

## Rollback

Run `deployctl rollback payments-api` to restore the previous build, which takes about five
minutes. Database migrations are not rolled back, so every migration must work with both the old
and the new build.
