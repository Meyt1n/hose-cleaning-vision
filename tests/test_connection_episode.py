from pathlib import Path
import sys,unittest,json,time
V = Path(__file__).resolve().parent
sys.path.insert(0, str(V.parent / 'project'))
from infer_service.engine.connection_episode import ConnectionEpisodeCounter

def feed(c,start,end,state='connected',step=100,epoch=0):
    out=[]
    for t in range(start,end+1,step):out.append(c.update(state,t,'1/1000',epoch=epoch))
    if (end-start)%step:out.append(c.update(state,end,'1/1000',epoch=epoch))
    return out

class Tests(unittest.TestCase):
    def test_short2p9(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,2900);self.assertEqual(c.confirmed_count,0)
    def test_short2p999(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,2999);self.assertEqual(c.confirmed_count,0)
    def test_exact3(self):
        c=ConnectionEpisodeCounter('A');r=feed(c,0,3000);self.assertEqual(c.confirmed_count,1);self.assertEqual(r[-1]['event']['duration'],3)
    def test_long_once(self):
        c=ConnectionEpisodeCounter('A');r=feed(c,0,10000);self.assertEqual(sum(x['event'] is not None for x in r),1)
    def test_discontinuous_no_sum(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,2000);c.update('disconnected',2040,'1/1000');feed(c,2080,4080);self.assertEqual(c.confirmed_count,0)
    def test_short_real_detach_splits(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,3000);c.update('disconnected',3040,'1/1000');feed(c,3080,6080);self.assertEqual(c.confirmed_count,2)
    def test_unknown_before_qualified_restarts(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,2000);c.update('unknown',2040,'1/1000');feed(c,2080,4980);self.assertEqual(c.confirmed_count,0);feed(c,5080,5080);self.assertEqual(c.confirmed_count,1)
    def test_unknown_after_qualified_not_duplicate(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,3000);c.update('unknown',3040,'1/1000');r=feed(c,3080,7080);self.assertEqual(c.confirmed_count,1);self.assertEqual(r[-1]['ambiguous_repeat_candidates'],1);self.assertTrue(r[-1]['review_required'])
    def test_repeated_unknowns_no_auto_new_event(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,3000)
        for start in [3080,7160,11240]:c.update('unknown',start,'1/1000');feed(c,start+80,start+4080)
        self.assertEqual(c.confirmed_count,1)
    def test_observed_detach_after_unknown_allows_new(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,3000);c.update('unknown',3040,'1/1000');c.update('disconnected',3080,'1/1000');feed(c,3120,6120);self.assertEqual(c.confirmed_count,2)
    def test_pts_gap_does_not_count_hidden_time(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,2000);r=c.update('connected',4000,'1/1000');self.assertEqual(r['continuous_seconds'],0);self.assertEqual(c.confirmed_count,0)
    def test_pts_gap_after_count_no_duplicate(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,3000);feed(c,5000,9000);self.assertEqual(c.confirmed_count,1)
    def test_pts_reversal(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,2000);r=c.update('connected',1900,'1/1000');self.assertEqual(r['connection_state'],'unknown');self.assertEqual(c.confirmed_count,0)
    def test_duplicate_pts_breaks_duration(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,2000);r=c.update('connected',2000,'1/1000');self.assertEqual(r['continuous_seconds'],0)
    def test_epoch_change(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,3000);r=feed(c,3100,7100,epoch=1);self.assertEqual(c.confirmed_count,1);self.assertTrue(r[-1]['review_required'])
    def test_invalid_pts(self):
        c=ConnectionEpisodeCounter('A');feed(c,0,2000);r=c.update('connected',None,'1/1000');self.assertEqual(r['connection_state'],'unknown');self.assertIsNone(c.run_start)
    def test_invalid_timebase(self):
        c=ConnectionEpisodeCounter('A');self.assertEqual(c.update('connected',0,'0')['connection_state'],'unknown')
    def test_action_logits_rejected(self):
        c=ConnectionEpisodeCounter('A')
        with self.assertRaises(ValueError):c.update('connected',0,'1/1000',evidence_scope='cleaning_action')
    def test_streams_separate(self):
        a=ConnectionEpisodeCounter('A');b=ConnectionEpisodeCounter('B');feed(a,0,3000);feed(b,0,2900);self.assertEqual((a.confirmed_count,b.confirmed_count),(1,0))
    def test_no_hose_claim(self):
        c=ConnectionEpisodeCounter('A');r=feed(c,0,3000)[-1];self.assertIsNone(r['event']['physical_hose_id']);self.assertFalse(r['event']['per_hose_completion'])
    def test_source_timebase90k(self):
        c=ConnectionEpisodeCounter('A');events=[]
        for t in range(0,270001,9000):events.append(c.update('connected',5228027430+t,'1/90000'))
        self.assertEqual(events[-1]['event']['duration'],3)


if __name__ == '__main__':
    unittest.main()
