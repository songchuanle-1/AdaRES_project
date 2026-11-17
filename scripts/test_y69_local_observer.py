"""Only the exact seven local Y69 additions can extend the keeper contract."""
import unittest
import observe_server2_with_local_y69 as observer


class LocalPreparationGuards(unittest.TestCase):
    def rows(self):
        return [['A', p] for p in observer.expected_additions('adares_y69_coarse_geometry_bias')]

    def test_exact_seven_new_paths(self):
        observer.validate_local_preparation(self.rows())

    def test_modified_keeper_extra_missing_or_duplicate_refused(self):
        rows=self.rows()
        for values in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]],
                       rows[:-1]+[['M','AdaRES/ipdn/model_paper5/adares_y68_language_mask_modulation/dec.py']]):
            with self.assertRaises(AssertionError):observer.validate_local_preparation(values)

    def test_modified_deleted_renamed_or_foreign_candidate_refused(self):
        for status in ('M','D','R100'):
            rows=self.rows();rows[0][0]=status
            with self.assertRaises(AssertionError):observer.validate_local_preparation(rows)
        rows=self.rows();rows[0][1]='AdaRES/configs_song/model_paper5/foreign.yaml'
        with self.assertRaises(AssertionError):observer.validate_local_preparation(rows)


if __name__=='__main__':unittest.main()
