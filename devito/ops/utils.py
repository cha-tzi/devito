from collections import OrderedDict


# OPS Conventions
namespace = OrderedDict()

# OPS Kernel strings.
namespace['ops_acc'] = 'OPS_ACC'
namespace['ops_init'] = 'ops_init'
namespace['ops_partition'] = 'ops_partition'
namespace['ops_timing_output'] = 'ops_timing_output'
namespace['ops_exit'] = 'ops_exit'
namespace['ops-define-dimension'] = lambda i: '#define OPS_%sD' % i
