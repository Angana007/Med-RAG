# Why I Built Med-RAG — and What It Means for Lucidus

*Angana Chakraborty · anganachakraborty05@gmail.com*

---

Rhea, Aakaanksh —

You asked to see a tangible AI demo. Here it is: **[live demo URL]** — click the four scenario buttons, then try to break it. This page is the *why* behind what you'll see.

**Why healthcare, when I'm applying to an EdTech company?** Because I wanted to practice building AI for a domain where being wrong has consequences. A medical assistant that invents a lab value and a tutor that invents a fact teach the same lesson: the hard problem in applied AI isn't generating fluent answers — it's earning the right to be trusted with someone's decisions. Healthcare just makes that bar impossible to ignore. Education, done seriously, has the same bar: a student can't fact-check their tutor.

**What the demo actually proves.** Not "I can build RAG" — retrieval plus generation is a weekend tutorial now. Three harder things:

*It reasons, it doesn't just retrieve.* The showcase queries have no single document containing the answer. The system connects a declining lab trend, a hospital dosing protocol, and a patient's self-reported painkiller use into one clinical warning — and separately, reads six months of visits and explains *why* a number rebounded, not just that it did. Swap the nouns and that's exactly the reasoning an AI tutor needs: connect this student's quiz history, the curriculum standard, and their own stated confusion into one useful recommendation.

*It's shipped, not demoed.* Behind the UI: API-key auth, rate limiting, UUID-traced structured logs, health and metrics endpoints, typed error handling, CI with 24 automated tests, and a benchmark suite scoring faithfulness and safety on every change. I built the boring parts because the boring parts are what let a two-person product team sleep at night.

*It treats refusal as a feature.* Ask it for a phone number, a billing balance, or an MRI read and it declines — before the model even runs, at zero cost. Every answer shows its sources, its confidence, its latency, and its price. I believe judgment-support tools owe their users that transparency, and I suspect you do too — it's the difference between AI a school district can adopt and AI it has to ban.

**The economics are the quiet headline.** Local small-model inference costs ~$0.001 per *thousand* queries versus ~$10 per thousand on a frontier API. I made that trade deliberately and instrumented it visibly, because at classroom scale the unit economics decide which features can exist at all. Cost-awareness isn't an optimization pass at the end — it's a design input.

**How this maps to Lucidus.** The architecture is domain-agnostic by design: patient records → student learning histories; clinical protocols → curriculum standards and pedagogy guidelines; "what should the care team watch for" → "what should this student review before Thursday." The retrieval layer, the safety layers, the evaluation harness, and the cost model all transfer unchanged. What I'd bring to Lucidus isn't a medical chatbot — it's a working template for *safe, inspectable, affordable* reasoning over a learner's own data, plus the engineering habits that made it real.

I built this to be tested, not watched. Break it, and let's talk about what you'd want it to become.

— Angana
