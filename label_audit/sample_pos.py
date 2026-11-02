"""Draw a stratified sample of 'pos' label cells (7 per finding, distinct non-gold studies)
and dump their reports for manual audit; also report PPV of 'pos' on the gold 58."""
import json, csv, random, collections, sys
csv.field_size_limit(10**8)
K=['acl','mcl','mm','lm','moa','loa','pfoa','eff','syn','bak','con','frx']
NAME=dict(zip(K,["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]))
lab={}
for l in open(sys.argv[1],encoding='utf-8'):
    d=json.loads(l); lab[d['StudyInstanceUID']]=d
rows=list(csv.DictReader(open(sys.argv[2],encoding='utf-8')))
rep={r['StudyInstanceUID']:r['Report'] for r in rows}
gold={r['StudyInstanceUID']:r for r in rows if r['ACL'].strip()!=''}
print('GOLD check of pos cells  finding: gold_pos/n_pos  (mean p)')
tp=fp=0
for k in K:
    c=[(int(gold[s][NAME[k]].strip()=='1'),lab[s][k+'_p']) for s in gold if lab[s][k+'_v']=='pos']
    a=sum(x for x,_ in c); tp+=a; fp+=len(c)-a
    print(f'  {NAME[k]:17s} {a}/{len(c)}  mean_p={sum(p for _,p in c)/max(len(c),1):.2f}')
print(f'  overall PPV {tp}/{tp+fp} = {tp/(tp+fp):.2f}')
rng=random.Random(11); used=set(); out=[]
for k in K:
    cand=[s for s in lab if lab[s][k+'_v']=='pos' and s not in used and s not in gold]
    rng.shuffle(cand)
    for s in cand[:7]:
        used.add(s); out.append((k,s))
with open(sys.argv[3],'w',encoding='utf-8') as f:
    for i,(k,s) in enumerate(out,1):
        d=lab[s]; others=' '.join(f"{x}={d[x+'_v']}" for x in K)
        f.write(f"### {i:02d} | FINDING={NAME[k]} | p={d[k+'_p']} | study=...{s[-10:]}\nother stances: {others}\n{rep[s].strip()}\n\n")
print('wrote',len(out),'cells')
