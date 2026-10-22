# Experiment 13 — CBAA (bundle length L = 1) consensus belief.
#
# WHAT THIS MESSAGE IS
# One negotiating agent's current allocation belief: for every declared task, who it currently
# believes holds that task and at what bid. Exchange of these vectors is the whole of the CBAA
# consensus step; nothing here is a commitment, an ownership claim or a handoff instruction. The
# handoff protocol (HandoffRequest/Response/Confirm/ConfirmAck/OwnershipEvent) is unchanged and still
# owns ownership; this message only decides *which candidate the owner's commit step should name*.
#
# WHY IT IS A NEW MESSAGE INSTEAD OF A NEW FIELD ON AN EXISTING ONE
# Adding a repeated-array block to HandoffRequest or HandoffResponse would change the shape of a
# message that Experiments 07/12/14 and the deployed nodes already build, and would put experiment
# fields in the production path. A dedicated topic adds a message type and a line to CMakeLists and
# touches no existing definition (brief §5: prefer the additive alternative).
#
# RELATIONSHIP TO HandoffHeader (design decision D5)
# header.transaction_id IS the episode id, format <target_id>@e<ownership_epoch>#<attempt>. There is
# deliberately no separate "episode_id" field: two identity fields on one message drift apart, and the
# transaction table, the FSM and this negotiator must agree on which attempt is being discussed. The
# header also carries ownership_epoch, which is what fences a belief from before an abort-and-reclaim.
# header.message_type is 9, allocated by exp13_cbba_adapter.MSG_TYPE_ALLOCATION_BELIEF — 1..8 are the
# handoff protocol's own types (HandoffHeader.msg:13-20). The constant is NOT added to
# HandoffHeader.msg: a shared definition that eight packages compile would change for a value that
# only this experiment produces, and message-type registries belong with the producer until a second
# consumer appears.
#
# SCHEMA VERSIONING
# schema_version is a field, not an assumption: 1 preceded ownership_epoch, 2 added it, 3 added
# agreement_established. A receiver built against a different schema refuses the payload
# (UNSUPPORTED_SCHEMA) rather than reinterpreting it, so a mixed build cannot silently agree with
# itself. The constant lives in cbba_negotiator.BELIEF_SCHEMA_VERSION.
#
# PARALLEL ARRAYS
# interface definition language has no map<string, ...>, so the winning-bid vector is three parallel
# arrays. The invariants below are enforced by exp13_cbba_adapter.belief_from_mapping, which is the
# only code that turns a payload into a ConsensusMessage:
#   len(task_ids) == len(target_ids) == len(winning_bids) == len(winning_agents)
#   task_ids entries are unique and name the sender's DECLARED task set, not only the tasks it bid on
#   an unassigned task is ("", 0.0); a non-zero bid with an empty winner is malformed
#
# agreement_established is the sender's own consensus verdict (AGREED) for this episode, for the
# receiver to record as an acknowledgement. It is a report, not a quorum, not a lease and not a
# commitment — see EXP13_DESIGN_DECISIONS_20261020_210721.md D3 for what the four levels of "agreed"
# mean and which one arms a transfer.

HandoffHeader header          # identity: transaction_id == episode id; ownership_epoch binds it

uint32 schema_version         # 3 = carries agreement_established; mismatched payloads are refused
string belief_id              # <episode_id>:<sender_agent_id>:<round_number>; report de-duplication key
uint32 round_number           # this sender's own round counter, strictly increasing per episode

string[] task_ids             # declared tasks, sorted, identical set on every participant
string[] target_ids           # target named per task (REQ-206: explicit, never inferred from index)
float64[] winning_bids        # best bid this agent knows for each task; 0.0 when unassigned
string[] winning_agents       # holder of that bid; "" when unassigned
string[] bundle               # this agent's own task bundle (L = 1, so at most one entry)

bool agreement_established    # sender reached AGREED for this episode at this belief
