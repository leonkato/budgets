import numpy as np

### some globals specific to this problem

LABELS = ['Function', 'Object_Type', 'Operating_Status', 'Position_Type',
          'Pre_K', 'Reporting', 'Sharing', 'Student_Type', 'Use']

BPCI = [slice(0, 37, None),  slice(37, 48, None), slice(48, 51, None), slice(51, 76, None),
        slice(76, 79, None), slice(79, 82, None), slice(82, 87, None), slice(87, 96, None), slice(96, 104, None)]

BOX_PLOTS_COLUMN_INDICES = BPCI

the_labels = {'Function': np.array(['Aides Compensation', 'Career & Academic Counseling', 'Communications',
                                 'Curriculum Development', 'Data Processing & Information Services',
                                 'Development & Fundraising', 'Enrichment', 'Extended Time & Tutoring',
                                 'Facilities & Maintenance', 'Facilities Planning',
                                 'Finance, Budget, Purchasing & Distribution', 'Food Services',
                                 'Governance', 'Human Resources',
                                 'Instructional Materials & Supplies', 'Insurance', 'Legal',
                                 'Library & Media', 'NO_LABEL', 'Other Compensation',
                                 'Other Non-Compensation', 'Parent & Community Relations',
                                 'Physical Health & Services', 'Professional Development',
                                 'Recruitment', 'Research & Accountability',
                                 'School Administration', 'School Supervision', 'Security & Safety',
                                 'Social & Emotional',  'Special Population Program Management & Support',
                                 'Student Assignment', 'Student Transportation', 'Substitute Compensation',
                                 'Teacher Compensation',  'Untracked Budget Set-Aside', 'Utilities']),
              'Object_Type': np.array(['Base Salary/Compensation', 'Benefits', 'Contracted Services',
                                       'Equipment & Equipment Lease', 'NO_LABEL',
                                       'Other Compensation/Stipend', 'Other Non-Compensation',
                                       'Rent/Utilities', 'Substitute Compensation', 'Supplies/Materials',
                                       'Travel & Conferences']),
              'Operating_Status': np.array(['Non-Operating', 'Operating, Not PreK-12', 'PreK-12 Operating']),
              'Position_Type': np.array(['(Exec) Director', 'Area Officers', 'Club Advisor/Coach',
                                         'Coordinator/Manager', 'Custodian', 'Guidance Counselor',
                                         'Instructional Coach', 'Librarian', 'NO_LABEL', 'Non-Position',
                                         'Nurse', 'Nurse Aide', 'Occupational Therapist', 'Other',
                                         'Physical Therapist', 'Principal', 'Psychologist',
                                         'School Monitor/Security', 'Sec/Clerk/Other Admin',
                                         'Social Worker', 'Speech Therapist', 'Substitute', 'TA', 'Teacher',
                                         'Vice Principal']),
              'Pre_K': np.array(['NO_LABEL', 'Non PreK', 'PreK']),
              'Reporting': np.array(['NO_LABEL', 'Non-School', 'School']),
              'Sharing': np.array(['Leadership & Management', 'NO_LABEL', 'School Reported',
                                   'School on Central Budgets', 'Shared Services']),
              'Student_Type': np.array(['Alternative', 'At Risk', 'ELL', 'Gifted', 'NO_LABEL', 'Poverty',
                                        'PreK', 'Special Education', 'Unspecified']),
              'Use': np.array(['Business Services', 'ISPD', 'Instruction', 'Leadership',
                               'NO_LABEL', 'O&M', 'Pupil Services & Enrichment',
                               'Untracked Budget Set-Aside'])}


# indices is the set of slices that correspond to the labels (actually they're targets, not labels)
# the_labels is a dictionary mapping target name to array of labels for this target
def flat_to_labels(probas, indices=BPCI, targets=np.array(LABELS), the_labels=the_labels):
    ''' takes an array of probabilities (m, 104)
        returns an array of predictions of labels'''
    # we need to output an array of labels, num_targets by num_rows
    num_cols = len(targets)
    # probably won't need this
    num_rows = probas.shape[0]
    # make a place to put the output arrays
    the_outputs = []
    for idx, targ in enumerate(targets):
        out_col = the_labels[targ][np.argmax(probas[:, BPCI[idx]], axis=1)].reshape(-1, 1)
        the_outputs.append(out_col)
    return np.concatenate(the_outputs, axis=1)
